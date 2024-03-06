
import logging
wandb_logger = logging.getLogger('wandb')
wandb_logger.setLevel(logging.WARNING)

import matplotlib
import matplotlib.pyplot as plt
import torch
import os
import wandb
import numpy as np
import json
import random
from copy import deepcopy
import shutil
import os

from data.src.dataLoaders import GrooveDataSet_Control
from .eval_vaeder_tools import *

# (nic) running from root with `python -m eval.Control.evaluate_vaeder_models`
# some paths below may need to be adjusted

DATASET_JSON_PATH = "../../data/dataset_json_settings/4_4_BeatsAndFills_gmd.json"
WANDB_PATH = "npb/ControlAdversarial/"

MODELS_DICT = {"absurd_pine_6": "240:v3"}

GENRE_JSON_PATH = "data/control/gmd_genre_dict.json"
SERIALIZE_WHOLE_MODEL = True
SERIALIZE_MODEL_COMPONENTS = False


"""
name, epoch/version data, model class, 
"""

if __name__ == "__main__":

    # Start wandb
    # run = wandb.init()
    # genre_dict = 0

    # load the models
    for model_name, artifact_path in MODELS_DICT.items():

        # print(f"\n**Downloading {model_name}")

        # path = WANDB_PATH + "model_epoch_" + artifact_path
        # artifact = run.use_artifact(path, type="model")
        # artifact_dir = artifact.download()
        # epoch = path.split("model_epoch_")[-1].split(":")[0]
        # model = load_vaeder_model(os.path.join(artifact_dir, f"{epoch}.pth"), genre_json_path=GENRE_JSON_PATH)
        model = load_vaeder_model("/Users/npb/Desktop/Neutone/GrooveTransformer/misc/Control_VAE/ControlAdversarial/240.pth")

        MODELS_DICT[model_name] = {"model": model}

    model_path = "models_eval"
    os.makedirs(model_path, exist_ok=True)
    os.chdir(model_path)

    # Serialize
    if SERIALIZE_WHOLE_MODEL:
        print("\nSerializing models")
        make_empty_folder("serialized_models")
        for model_name, model_data in MODELS_DICT.items():
            model_data['model'].serialize_whole_model(model_name, os.getcwd())

        os.chdir("../")

    if SERIALIZE_MODEL_COMPONENTS:
        print("\nSerializing model components")
        make_empty_folder("serialized_components")
        for model_name, model_data in MODELS_DICT.items():

            model_data['model'].serialize(model_name)
        os.chdir("../")




