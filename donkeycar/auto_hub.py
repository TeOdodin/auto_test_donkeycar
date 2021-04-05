from auto_training import auto_train
from auto_test import auto_test

model_types = ["custom"]
datasets = ["data"]

auto_train(model_types, datasets)
for model_type in model_types:
    for dataset in datasets:
        model_path = f"models/model_{model_type}_{dataset}.h5"
        auto_test(model_type, model_path, dataset)