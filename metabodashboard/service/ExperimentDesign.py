DATA_MATRIX = "Data\\Matrix_normalised_pos.csv"

# Cas, Temoin, TC
EXPERIMENT_DESIGNS = {
    "Ctrl_vs_Case": {
        "classes": {"Controls": ["Temoin"], "Cases": ["Cas"]},
        "TestSize": 0.2,
    },
    # "Control vs TC":{
    #     "classes": {
    #         "Controles":["Cas"],
    #         "TC": ["TC"]
    #     },
    #     "TestSize": 0.2,
    # },
    # "TC vs Cas":{
    #     "classes": {
    #         "TC":["TC"],
    #         "Cases": ["Cas"]
    #     },
    #     "TestSize": 0.2,
    # },
    # "Control vs all":{
    #     "classes": {
    #         "Control":["Temoin"],
    #         "All": ["Cas", "TC"]
    #     },
    #     "TestSize": 0.2,
    # },
}
