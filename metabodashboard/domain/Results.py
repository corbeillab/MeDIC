from typing import List

import numpy as np
import pandas as pd
from abc import abstractmethod
import os
import pickle

import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, \
    recall_score, f1_score, roc_auc_score, balanced_accuracy_score

from collections import Counter
import umap
from sklearn.decomposition import PCA

from . import MetaData
from ..service import Utils

ROOT_PATH = os.path.dirname(__file__)
DUMP_PATH = os.path.join(ROOT_PATH, os.path.join("dumps", "splits"))

class Results:
    """
    Contains all results of an experimental design, is an attribute of class Experimental_design, and gives info to class "Plotter".
    Has results of all algorithms for all splits on one experimental design (so almost only numbers/floats/ints).
    Can be kept in RAM as it is not supposed to be too big, and prevents the reading/writing of models and splits files.
    """

    def __init__(self, splits_number: int):
        self.splits_number = [str(s) for s in range(splits_number)]
        self.results = {s: {} for s in self.splits_number}
        self.f_names = []
        self.best_acc = 0
        self.design_name = ""

    @abstractmethod
    def _get_features_importance(self, model):
        """
        retrieve features and their importance from a model to save it in the Results dict after each split
        """
        raise NotImplementedError()

    @abstractmethod
    def _aggregate_features_info(self):
        """
        When all splits are done and saved, aggregate feature info from every split to compute stats
        from all splits, concatenate in the same list the name of features, and another list their importance
        """
        raise NotImplementedError()

    def add_results_from_one_algo_on_one_split(self, model: sklearn, scaled_data: pd.DataFrame, classes: list,
                                               y_train_true: list, y_train_pred: list,
                                               y_test_true: list, y_test_pred: list, split_number: str, train_ids: List[str], test_ids: List[str]):
        """
        Besoin modèle pour extraire features, features importance
        Besoin des y_true, des y_pred, des noms de samples pour le train et le test

        Fonction appelée à chq fois qu'un split a fini de rouler, pour stocker les info nécessaires à la production des
        graphique pour l'onglet résultat
        X : entièreté du dataset (autant train que test) c'est simplement pour voir le clustering de tous les individus
        """
        self.results[split_number]["y_test_true"] = y_test_true
        self.results[split_number]["y_test_pred"] = y_test_pred
        self.results[split_number]["y_train_true"] = y_train_true
        self.results[split_number]["y_train_pred"] = y_train_pred
        self.results[split_number]["train_accuracy"] = accuracy_score(y_train_true, y_train_pred)
        self.results[split_number]["test_accuracy"] = accuracy_score(y_test_true, y_test_pred)
        self.results[split_number]["balanced_train_accuracy"] = balanced_accuracy_score(y_train_true, y_train_pred)
        self.results[split_number]["balanced_test_accuracy"] = balanced_accuracy_score(y_test_true, y_test_pred)
        binary_y_train_true = Utils.get_binary(y_train_true, classes)
        binary_y_train_pred = Utils.get_binary(y_train_pred, classes)
        binary_y_test_true = Utils.get_binary(y_train_true, classes)
        binary_y_test_pred = Utils.get_binary(y_train_pred, classes)
        self.results[split_number]["train_precision"] = precision_score(binary_y_train_true, binary_y_train_pred)
        self.results[split_number]["test_precision"] = precision_score(binary_y_test_true, binary_y_test_pred)
        self.results[split_number]["train_recall"] = recall_score(binary_y_train_true, binary_y_train_pred)
        self.results[split_number]["test_recall"] = recall_score(binary_y_test_true, binary_y_test_pred)
        self.results[split_number]["train_f1"] = f1_score(binary_y_train_true, binary_y_train_pred)
        self.results[split_number]["test_f1"] = f1_score(binary_y_test_true, binary_y_test_pred)
        self.results[split_number]["train_roc_auc"] = roc_auc_score(binary_y_train_true, binary_y_train_pred)
        self.results[split_number]["test_roc_auc"] = roc_auc_score(binary_y_test_true, binary_y_test_pred)
        self.results[split_number]["failed_samples"] = self.produce_always_wrong_samples(y_train_true, y_train_pred,
                                                                                         y_test_true, y_test_pred,
                                                                                         split_number, train_ids, test_ids)
        if self.results[split_number]["test_accuracy"] > self.best_acc:
            self.best_acc = self.results[split_number]["test_accuracy"]
            self.results["best_model"] = model
        self.results[split_number]["feature_importances"] = self._get_features_importance(model)
        self.results[split_number]["Confusion_matrix"] = self._produce_conf_matrix(y_test_true, y_test_pred)

        if split_number == self.splits_number[-1]:
            self.results["info_expe"] = self._produce_info_expe(y_train_true, y_test_true)
            print("------> last split, start features importance")
            self.results["features_table"] = self.produce_features_importance_table()
            self.results["accuracies_table"] = self.produce_accuracy_plot_all()
            self.results["classes"] = classes
            self.results["umap_data"] = self._produce_UMAP(scaled_data, self.results["features_table"])
            self.results["pca_data"] = self._produce_PCA(scaled_data, self.results["features_table"])
            self.results["metrics_table"] = self.produce_metrics_table()
            self.results["features_stripchart"] = self.features_strip_chart_abundance_each_class(self.results["features_table"], scaled_data)

    def set_feature_names(self, x: pd.DataFrame):
        """
        retrieve features name directly from datamatrix
        """
        self.f_names = list(x.columns)

    def format_name_and_associated_values(self, names, values):
        """
        from a Counter dict, modify
        """
        count = Counter(names)
        for n in count.keys():
            count[n] = [0]
            liste_val = []
            for idx, j in enumerate(names):
                if n == j and values[idx] > 0:
                    count[n][0] += 1
                    liste_val.append(values[idx])
            if liste_val:
                count[n].append(np.mean(liste_val))
            else:
                count[n].append(0)
        return count

    def _produce_conf_matrix(self, y_test_true: list, y_test_pred: list):
        labels = list(set(y_test_true))
        return labels, confusion_matrix(y_test_true, y_test_pred, labels=labels, normalize="true")

    def _produce_UMAP(self, X: pd.DataFrame, features_df: pd.DataFrame):
        nbr_feat = [10, 40, 100]
        umaps = []
        for nbr in nbr_feat:
            selected_feat = features_df["features"][:nbr]
            selected_x = X.loc[:, selected_feat]
            print(list(zip(selected_x.index, self.results["classes"])))
            selected_x = selected_x.to_numpy()
            umap_data = umap.UMAP(n_components=2, init='random', random_state=13)
            umaps.append(umap_data.fit_transform(selected_x))
        # Redo the umap but on all the data
        selected_x = X.to_numpy()
        umap_data = umap.UMAP(n_components=2, init='random', random_state=13)
        umaps.append(umap_data.fit_transform(selected_x))
        return umaps

    def _produce_PCA(self, X: pd.DataFrame, features_df: pd.DataFrame):
        nbr_feat = [10, 40, 100]
        pcas = []

        for nbr in nbr_feat:
            selected_feat = features_df["features"][:nbr]
            x = X.loc[:, selected_feat]
            x = x.to_numpy()

            pca = PCA(n_components=2)
            pcas.append(pca.fit_transform(x))
        # Redo the umap but on all the data
        x = X.to_numpy()
        pca = PCA(n_components=2)
        pcas.append(pca.fit_transform(x))
        return pcas

    def _produce_info_expe(self, y_train_true, y_test_true):
        """
        produce dataframe with basic information about the dataset/experiment, like number of samples and the train-test
        proprotion, the number of class, etc.
        """
        nbr_train = len(y_train_true)
        nbr_test = len(y_test_true)
        tot = nbr_train + nbr_test
        nom_stats = ["Number of samples", "Train-test repartition"]
        valeurs_stats = [str(tot)]
        valeurs_stats.append(str(int(nbr_train / tot * 100)) + " - " + str(int(nbr_test / tot * 100)))
        y = y_train_true + y_test_true
        c = Counter(y)
        for k in c.keys():
            nom_stats.append("Number of class {}".format(k))
            valeurs_stats.append("{}".format(c[k]))

        d = {"stats": nom_stats, "numbers": valeurs_stats}
        df = pd.DataFrame(data=d)
        return df

    def produce_features_importance_table(self):
        """
        Fonction qui réccupère les features et leurs importances de chq split sur le train et le test pour en faire un dataframe.
        Est donnée à la fonction de plotting correspondante (après que l'instance ait été complétée avec tous
        les résultats de splits)
        """
        features, times_used_all_splits, importance_or_usage_or_ = self._aggregate_features_info()
        print("--> aggregating done, importances : {}".format(importance_or_usage_or_))

        d = {"features": features, "times_used": times_used_all_splits, "importance_usage": importance_or_usage_or_}
        df = pd.DataFrame(data=d)
        df["times_used"] = pd.to_numeric(df["times_used"])
        df["importance_usage"] = pd.to_numeric(df["importance_usage"])
        df = df.sort_values(by=['importance_usage'], ascending=False)
        return df

    def produce_accuracy_plot_all(self):
        """
        Fonction qui réccupère les résultats (accuracies) de chq split sur le train et le test pour en faire un dataframe.
        Est donnée à la fonction de plotting correspondante (après que l'instance ait été complétée avec tous
        les résultats de splits)
        """
        x_splits_num = []
        y_splits_acc = []
        traces = []
        for s in self.splits_number:
            x_splits_num.append(str(s))  # c'est normal
            x_splits_num.append(str(s))
            y_splits_acc.append(self.results[s]["train_accuracy"])
            traces.append("train")
            y_splits_acc.append(self.results[s]["test_accuracy"])
            traces.append("test")

        d = {"splits": x_splits_num, "accuracies": y_splits_acc, "color": traces}
        df = pd.DataFrame(data=d)

        return df

    # TODO: faire une fonction qui produce metrics table pour tous les splits
    def produce_metrics_table(self):
        metrics = ["accuracy", "balanced accuracy", "precision", "recall", "f1", "roc_auc"]
        trains_metrics = []
        tests_metrics = []
        acctrain = []
        acctest = []
        balacctrain = []
        balacctest = []
        precisiontrain = []
        precisiontest = []
        recalltrain = []
        recalltest = []
        f1train = []
        f1test = []
        roc_auc_train = []
        roc_auc_test = []
        for s in self.splits_number:
            acctrain.append(self.results[s]["train_accuracy"])
            acctest.append(self.results[s]["test_accuracy"])
            balacctrain.append(self.results[s]["balanced_train_accuracy"])
            balacctest.append(self.results[s]["balanced_test_accuracy"])
            precisiontrain.append(self.results[s]["train_precision"])
            precisiontest.append(self.results[s]["test_precision"])
            recalltrain.append(self.results[s]["train_recall"])
            recalltest.append(self.results[s]["test_recall"])
            f1train.append(self.results[s]["train_f1"])
            f1test.append(self.results[s]["test_f1"])
            roc_auc_train.append(self.results[s]["train_roc_auc"])
            roc_auc_test.append(self.results[s]["test_roc_auc"])

        trains_metrics.append(
            str(round(float(np.mean(acctrain)), 4)) + " (" + str(round(float(np.std(acctrain)), 4)) + ")")
        trains_metrics.append(
            str(round(float(np.mean(balacctrain)), 4)) + " (" + str(round(float(np.std(balacctrain)), 4)) + ")")
        trains_metrics.append(
            str(round(float(np.mean(precisiontrain)), 4)) + " (" + str(round(float(np.std(precisiontrain)), 4)) + ")")
        trains_metrics.append(
            str(round(float(np.mean(recalltrain)), 4)) + " (" + str(round(float(np.std(recalltrain)), 4)) + ")")
        trains_metrics.append(
            str(round(float(np.mean(f1train)), 4)) + " (" + str(round(float(np.std(f1train)), 4)) + ")")
        trains_metrics.append(
            str(round(float(np.mean(roc_auc_train)), 4)) + " (" + str(round(float(np.std(roc_auc_train)), 4)) + ")")

        tests_metrics.append(
            str(round(float(np.mean(acctest)), 4)) + " (" + str(round(float(np.std(acctest)), 4)) + ")")
        tests_metrics.append(
            str(round(float(np.mean(balacctest)), 4)) + " (" + str(round(float(np.std(balacctest)), 4)) + ")")
        tests_metrics.append(
            str(round(float(np.mean(precisiontest)), 4)) + " (" + str(round(float(np.std(precisiontest)), 4)) + ")")
        tests_metrics.append(
            str(round(float(np.mean(recalltest)), 4)) + " (" + str(round(float(np.std(recalltest)), 4)) + ")")
        tests_metrics.append(str(round(float(np.mean(f1test)), 4)) + " (" + str(round(float(np.std(f1test)), 4)) + ")")
        tests_metrics.append(
            str(round(float(np.mean(roc_auc_test)), 4)) + " (" + str(round(float(np.std(roc_auc_test)), 4)) + ")")

        metrics_table = pd.DataFrame(data={"metrics": metrics, "train": trains_metrics, "test": tests_metrics})
        return metrics_table

    def features_strip_chart_abundance_each_class(self, feature_df, data):
        """
        store data for the 10 most important feature (mean of all split)
        as well as the class for each sample
        allows ploting the stripchart in Plots
        """
        important_features = list(feature_df["features"])[:10]
        df = data.loc[:, important_features]
        print(df)
        print(len(self.results["classes"]))
        df["targets"] = self.results["classes"]
        return df

    def produce_always_wrong_samples(self, y_train_true, y_train_pred, y_test_true, y_test_pred, split_number,
                                     train_ids: List[str], test_ids: List[str]):
        """
        return: two dicts with sample names as keys, and wrongly predicted as values (0:good pred, 1:bad pred)
        """

        train_samples = {t: 0 for t in train_ids}
        test_samples = {t: 0 for t in test_ids}

        labels = {l: idx for idx, l in enumerate(list(set(y_train_true)))}


        y_train_true = [labels[l] for l in y_train_true]
        y_train_pred = [labels[l] for l in y_train_pred]
        y_test_true = [labels[l] for l in y_test_true]
        y_test_pred = [labels[l] for l in y_test_pred]

        train_nbr = [sum(x) for x in list(zip(y_train_true, y_train_pred))]
        for i, n in enumerate(train_ids):
            if train_nbr[i] == 1:
                train_samples[n] += 1

        test_nbr = [sum(x) for x in list(zip(y_test_true, y_test_pred))]
        for i, n in enumerate(test_ids):
            if test_nbr[i] == 1:
                test_samples[n] += 1

        return train_samples, test_samples


class ResultsDT(Results):
    """
    Contains all results of an experimental design, is an attribute of class Experimental_design, and gives info to class "Plotter".
    Has results of all algorithms for all splits on one experimental design (so almost only numbers/floats/ints).
    Can be kept in RAM as it is not supposed to be too big, and prevents the reading/writing of models and splits files.
    """

    def _get_features_importance(self, model):
        """
        retrieve features and their importance from a model to save it in the Results dict after each split
        """
        if self.f_names is None:
            raise RuntimeError("Features names are not retrieved yet")
        print("----> entered in _get_features_importance of DT, importances :")
        importances = model.feature_importances_
        zipped = zip(self.f_names, importances)
        return zipped

    def _aggregate_features_info(self):
        """
        When all splits are done and saved, aggregate feature info from every split to compute stats
        from all splits, concatenate in the same list the name of features, and another list their importance
        """
        features = []
        imp = []
        # Get values of all splits in two lists
        for split in self.splits_number:
            f, i = list(zip(*self.results[split]["feature_importances"]))
            features.extend(f)
            imp.extend(i)

        # Store the mean importance, and the number of time used, per feature
        count_f = self.format_name_and_associated_values(features, imp)

        features = [f for f in count_f.keys()]
        times_used_all_splits = [count_f[f][0] for f in count_f.keys()]
        importance_or_usage_or_ = [count_f[f][1] for f in count_f.keys()]
        return features, times_used_all_splits, importance_or_usage_or_


class ResultsRF(Results):
    """
    Contains all results of an experimental design, is an attribute of class Experimental_design, and gives info to class "Plotter".
    Has results of all algorithms for all splits on one experimental design (so almost only numbers/floats/ints).
    Can be kept in RAM as it is not supposed to be too big, and prevents the reading/writing of models and splits files.
    """

    def _get_features_importance(self, model):
        if self.f_names is None:
            raise RuntimeError("Features names are not retrieved yet")

        # features = []
        # importances = []
        # for DT in model.estimators_:
        #     i = DT.feature_importances_
        #     zipped = list(zip(self.f_names, i))
        #     feat_sort = sorted(zipped, key=lambda x: x[1], reverse=True)
        #     top_x = feat_sort[:50]
        #     f, i = zip(*top_x)
        #     features.extend(f)
        #     importances.extend(i)
        # zipped = zip(features, importances)

        importances = model.feature_importances_

        # importances = model.feature_importances_
        zipped = zip(self.f_names, importances)
        # zipped_complet = zip(model.feature_names_in_, model.feature_importances_)
        return zipped  # , zipped_complet

    def _aggregate_features_info(self):
        """
        When all splits are done and saved, aggregate feature info from every split to compute stats
        from all splits, concatenate in the same list the name of features, and another list their importance
        """
        features = []
        imp = []
        # features_complet = []
        # imp_complet = []
        # Get values of all splits in two lists
        for split in self.splits_number:
            f, i = list(zip(*self.results[split]["feature_importances"]))
            features.extend(f)
            imp.extend(i)
            # f_complet, i_complet = zip(*self.results[split]["feature_importances"][1])
            # features_complet.extend(f_complet)
            # imp_complet.extend(i_complet)

        # Store the mean importance, and the number of time used, per feature
        dict_top = self.format_name_and_associated_values(features, imp)
        # dict_complet = self.format_name_and_associated_values(features_complet, imp_complet)

        # Top 5 of sub-classifier (DT) for features, and times_used
        # Top 5 of sub-classifier (DT) for importance_(mean global importance in RF)
        features = [f for f in dict_top.keys()]
        times_used_all_splits = [dict_top[f][0] for f in dict_top.keys()]
        importance_or_usage_or_ = [str(dict_top[f][1]) for f in
                                   dict_top.keys()]  # + "_(" + str(dict_complet[f][1]) + ")"
        return features, times_used_all_splits, importance_or_usage_or_


class ResultsSCM(Results):
    def _get_features_importance(self, model):
        if self.f_names is None:
            raise RuntimeError("Features names are not retrieved yet")

        features = pd.DataFrame(0, index=self.f_names, columns=["importance"])
        number_of_rules = 0
        for rule in model.model_.rules:
            feature_name = self.f_names[rule.feature_idx]
            features.loc[feature_name, "importance"] += 1
            number_of_rules += 1

        features["importance"] = features["importance"] / number_of_rules
        features.sort_values(by="importance", ascending=False, inplace=True)

        zipped = zip(features.index.values.tolist(), features["importance"].values)
        return zipped

    def _aggregate_features_info(self):
        features = []
        importances = []

        for split in self.splits_number:
            f, i = zip(*self.results[split]["feature_importances"])
            features.extend(f)
            importances.extend(i)

        dict_top = self.format_name_and_associated_values(features, importances)
        features = [f for f in dict_top.keys()]
        times_used_all_splits = [dict_top[f][0] for f in dict_top.keys()]
        importance_or_usage_or_ = [str(dict_top[f][1]) for f in
                                   dict_top.keys()]  # + "_(" + str(dict_complet[f][1]) + ")"
        return features, times_used_all_splits, importance_or_usage_or_


class ResultsRSCM(Results):
    def _get_features_importance(self, model):
        if self.f_names is None:
            raise RuntimeError("Features names are not retrieved yet")

        features = pd.DataFrame(0, index=self.f_names, columns=["importance"])
        number_of_rules = 0
        for estimator in model.get_estimators():
            for rule in estimator.model_.rules:
                feature_name = self.f_names[rule.feature_idx]
                features.loc[feature_name, "importance"] += 1
                number_of_rules += 1

        features["importance"] = features["importance"] / number_of_rules
        features.sort_values(by="importance", ascending=False, inplace=True)

        zipped = zip(features.index.values.tolist(), features["importance"].values)
        return zipped

    def _aggregate_features_info(self):
        features = []
        importances = []

        for split in self.splits_number:
            f, i = zip(*self.results[split]["feature_importances"])
            features.extend(f)
            importances.extend(i)

        dict_top = self.format_name_and_associated_values(features, importances)
        features = [f for f in dict_top.keys()]
        times_used_all_splits = [dict_top[f][0] for f in dict_top.keys()]
        importance_or_usage_or_ = [str(dict_top[f][1]) for f in
                                   dict_top.keys()]  # + "_(" + str(dict_complet[f][1]) + ")"
        return features, times_used_all_splits, importance_or_usage_or_