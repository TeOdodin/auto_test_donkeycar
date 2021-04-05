import donkeycar as dk
from donkeycar.pipeline.training import train

def auto_train(model_types, datasets):
    cfg = dk.load_config()

    for model_type in model_types:
        for dataset in datasets:
            model = f"models/model_{model_type}_{dataset}.h5"
            train(cfg, f"datasets/{dataset}", model, model_type)