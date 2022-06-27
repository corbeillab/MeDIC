from pyscm.scm import SetCoveringMachineClassifier
from randomscm.randomscm import RandomScmClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

LEARN_CONFIG = {
    "DecisionTree": {
        "function": DecisionTreeClassifier,
        "ParamGrid": {
            "max_depth": [1, 2, 3, 4, 5, 10],
            "min_samples_split": [2, 4, 6, 8, 10]
        }
    },
    "RandomForest": {
        "function": RandomForestClassifier,
        "ParamGrid": {
            "n_estimators": [1, 2, 4, 10, 30, 70, 100, 500, 1000]
        }
    },
    "SCM": {
        "function": SetCoveringMachineClassifier,
        "ParamGrid": {
            "p": [0.5, 1., 2.],
            "max_rules": [1, 2, 3, 4, 5],
            "model_type": ["conjunction", "disjunction"]
        }
    },
    "RandomSCM": {
        "function": RandomScmClassifier,
        "ParamGrid": {
            "p_options": [[0.5, 1., 2.]],
            "max_rules": [1, 2, 3, 4, 5],
            "model_type": ["conjunction", "disjunction"],
        }
    },
}
