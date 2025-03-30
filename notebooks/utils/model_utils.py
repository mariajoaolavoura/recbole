import importlib
from recbole.utils.enum_type import ModelType

import numpy as np
import torch
# from recbole.model.general_recommender.bpr import BPR
# from recbole.config import Config
# from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction



def get_trainer(model_type, model_name):
    r"""Copy of recbole.utils.utils but calls CustomTrainer
    
    Automatically select trainer class based on model type and model name

    Args:
        model_type (ModelType): model type
        model_name (str): model name

    Returns:
        Trainer: trainer class
    """
    try:
        print('in model_utils.get_trainer utils.custom_trainer.CustomTrainer')
        return getattr(
            importlib.import_module("utils.custom_trainer"), "CustomTrainer" )
    except AttributeError:
        if model_type == ModelType.KNOWLEDGE:
            return getattr(importlib.import_module("recbole.trainer"), "KGTrainer")
        elif model_type == ModelType.TRADITIONAL:
            return getattr(
                importlib.import_module("recbole.trainer"), "TraditionalTrainer"
            )
        else:
            print('in model_utils.get_trainer recbole.trainer.Trainer')
            return getattr(importlib.import_module("recbole.trainer"), "Trainer")