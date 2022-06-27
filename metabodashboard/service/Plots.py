import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import html
import umap


class Plots():
    def __init__(self, colors: str):
        self.colors = colors

    # TODO : faire la sauvegarde dans results des resultats de heatmap pour pouvoir sortir la figure
    def show_algo_comparison_by_heatmap(self):
        return

    def show_umap(self, umap_data, classes):
        fig = px.scatter(
            umap_data, x=0, y=1,
            color=classes,
            color_continuous_scale=self.colors,
            title=""
        )

        fig.update_layout({
            "plot_bgcolor": "rgba(0, 0, 0, 0)",
            "paper_bgcolor": "rgba(0, 0, 0, 0)",
        },
            title="UMAP applied on top X features selected by the algorithm"
        )
        return fig

    def show_PCA(self, pca_data, classes):
        fig = px.scatter(pca_data, x=0, y=1,
                         color=classes,
                         color_continuous_scale=self.colors,
                         )
        fig.update_layout({
            "plot_bgcolor": "rgba(0, 0, 0, 0)",
            "paper_bgcolor": "rgba(0, 0, 0, 0)",
        },
            title="PCA applied on top X features selected by the algorithm"
        )
        return fig

    def show_general_confusion_matrix(self, cm, labels, text):
        # labels = ["0", "1"]

        fig = go.Figure(data=go.Heatmap(
            # labels=dict(x="Prediciton", y="Vérité", color="Nombre de prédictions"),
            z=cm,
            x=labels,
            y=labels,
            # text=text,
            colorscale=self.colors,
            showscale=False
            # texttemplate="%{text}",
        ))
        fig = fig.update_traces(text=text, texttemplate="%{text}", hovertemplate=None)
        fig.update_layout(xaxis_title="Prediciton",
                          yaxis_title="Truth",

                          )

        # fig = px.imshow(
        #         cm,
        #         labels=dict(x="Prediciton", y="Vérité", color="Nombre de prédictions"),
        #         x=list(set(labels)),
        #         y=list(set(labels)),
        #         color_continuous_scale=self.colors,
        #         text_auto=True
        # )
        # fig.update_traces(text=text)
        return fig

    def show_accuracy_all(self, df):
        """
        plot the accuracy for each split on train and test set
        df : generated from Results.produce_accuracy_plot_all()
        """
        if "splits" not in df.columns:
            raise RuntimeError("To show the global accuracies plot, the dataframe needs to have a 'splits' column")
        if "accuracies" not in df.columns:
            raise RuntimeError("To show the global accuracies plot, the dataframe needs to have a 'accuracies' column")
        if "color" not in df.columns:
            raise RuntimeError("To show the global accuracies plot, the dataframe needs to have a 'color' column")

        fig = px.line(df, x='splits', y='accuracies', color='color', title="Accuracies on train and test sets for each split")
        fig.update_yaxes(range=[0, 1.1])
        fig.update_layout({
            "plot_bgcolor": "rgba(246, 247, 247, 0.4)",
            "paper_bgcolor": "rgba(0, 0, 0, 0)"
        })
        return fig

    def show_exp_info_all(self, df: pd.DataFrame):
        """
        display in table the number of samples, per classes, in train/test, etc.
        df : generated from Results.produce_info_expe()
        """
        if "stats" not in df.columns:
            raise RuntimeError("To show the global accuracies plot, the dataframe needs to have a 'stats' column")
        if "numbers" not in df.columns:
            raise RuntimeError("To show the global accuracies plot, the dataframe needs to have a 'numbers' column")

        # fig = go.Figure(
        #     data=[go.Table(
        #         cells=dict(values=[df.stats, df.numbers]))
        #         ])

        row1 = html.Tr([html.Td(df.iloc[0, 0]), html.Td(df.iloc[0, 1])])
        row2 = html.Tr([html.Td(df.iloc[1, 0]), html.Td(df.iloc[1, 1])])
        row3 = html.Tr([html.Td(df.iloc[2, 0]), html.Td(df.iloc[2, 1])])
        table_body = [html.Tbody([row1, row2, row3])]
        return table_body

    def show_features_selection(self, df: pd.DataFrame):
        """
        table of features used by all models (all split of an algorithm)
        ranked by most used first
        only display 10 most or used in at least 75%? of models ?
        df : generated from Results.produce_features_importance_table()
        """
        if "features" not in df.columns:
            raise RuntimeError("To show the global accuracies plot, the dataframe needs to have a 'features' column")
        if "times_used" not in df.columns:
            raise RuntimeError("To show the global accuracies plot, the dataframe needs to have a 'times_used' column")
        if "importance_usage" not in df.columns:
            raise RuntimeError(
                "To show the global accuracies plot, the dataframe needs to have a 'importance_usage' column")
        # TODO : sort data by times_used or importance, and take only top 10-20 to display

        fig = go.Figure(
            data=[go.Table(
                header=dict(values=list(df.columns), align="center"),
                cells=dict(values=[df.iloc[:10, :].features, df.iloc[:10, :].times_used,
                                   df.iloc[:10, :].importance_usage],
                           align="center"))
            ])
        fig.update_layout(
            title="Table of top 10 features sorted by importance"
        )
        return fig

    def show_split_metrics(self):
        """
        display in table the number of samples, per classes, in train/test, etc. for one split
        """
        return

    def show_metabolite_levels(self, features_data, feature):
        """
        Plot in stripchart (boxplot with point and no box)
        (with a dropdown to select the metabolite, max of N? metabolite)
        And show the intensity of this metabolite/ this feature in each class (one box per class)
        """
        df = features_data
        fig = px.strip(df, x='targets', y=feature, title="Abundance of metabolite {} in each sample by class".format(feature))
        return fig

    def show_heatmap_wrong_samples(self, data_train, data_test, samples_names, algos):
        fig = go.Figure()
        # fig.add_trace(
        #     go.Heatmap(
        #         z=data_train,
        #         x=algos,
        #         y=samples_names,
        #         opacity=1,
        #         colorscale="Blues"
        #     )
        # )
        fig.add_trace(
            go.Heatmap(
                z=data_test,
                x=algos,
                y=samples_names,
                opacity=1,
                colorscale="Reds"
            )
        )

        return fig