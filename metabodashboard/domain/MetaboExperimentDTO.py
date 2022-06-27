from . import MetaboExperiment


class MetaboExperimentDTO:
    def __init__(self, metabo_experiment: MetaboExperiment):
        self.data_matrix = metabo_experiment.get_data_matrix()
        self.metadata = metabo_experiment.get_metadata()

        self.number_of_splits = metabo_experiment.get_number_of_splits()
        self.train_test_proportion = metabo_experiment.get_train_test_proportion()

        self.experimental_designs = metabo_experiment.get_experimental_designs()

        self.custom_models = metabo_experiment.get_custom_models()
        self.selected_models = metabo_experiment.get_selected_models()

        self.selected_cv_type = metabo_experiment.get_selected_cv_type()
