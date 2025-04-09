#################################################################################################################
#### Imports

from .file_utils import validate_and_save_picklefile

#### Recbole dataset configuration functions
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, init_logger
from logging import getLogger


#### Recbole Train and Evaluate functions
import importlib
from recbole.utils.enum_type import ModelType
import numpy as np
import torch
from recbole.data.interaction import Interaction
from collections import OrderedDict
from .itemknn import ItemKNN
from recbole.model.general_recommender import BPR, Pop, NeuMF



#################################################################################################################
#### Filenames functions

def get_folderpath_str(save_path, base_filename, specs_str):
    return save_path+base_filename+'_'+specs_str+'/'

def get_evaluation_results_filename(model_name, model_part, section, filename_version=''):
    if section == 'diagonal' and model_part is None:
        return model_name+'_evaluation_results_diagonal'+filename_version
    elif model_part[-3:][:2] == 'pt': 
        return model_name+'_evaluation_results_model_'+model_part[-3:]+'_section_'+section[-3:]+filename_version
    else:
        return model_name+'_evaluation_results_model_full_section_'+section[-3:]+filename_version


def get_test_full_data_sections(model_version:str, 
                            models_versions=['_pt1', '_pt2', '_pt3', '']):
    """
        model_version:str ['_pt1', '_pt2', '_pt3', '']

        freq=3 # 3 month
        duration = 2*12//freq # 2 years split in 3M buckets
        n_parts = duration*2+1

            Dataset section parts
                |    part 8    |
        |part 1|part 5| part 6|part 7|
        |    part 2   |
        |         part 3      |
        |             ''             |
    """
    
    # duration = len(models_versions)
    # n_parts = duration*2+1
    # triangle_holdouts = [models_versions[0]]+['_pt'+str(i) for i in range(duration+1, n_parts)]

    for i, pt in enumerate(models_versions):
        if model_version==pt:
            return models_versions[:i]+models_versions[i+1:]+['_pt8']
        

def get_test_full_data_sections_with_names(model_version:str, 
                                      base_dataset_name:str,
                                      models_versions=['_pt1', '_pt2', '_pt3', '']):
    """
        model_version:str ['_pt1', '_pt2', '_pt3', '']
            Dataset section parts
                |    part 8    |
        |part 1|part 5| part 6|part 7|
        |    part 2   |
        |         part 3      |
        |             ''             |
    """    
    test_datasec = get_test_full_data_sections(model_version, models_versions)

    return [base_dataset_name+datasec for datasec in test_datasec]


#################################################################################################################
#### Recbole dataset configuration functions

def setup_config_and_dataset(model_name,
                            dataset_name,
                            parameter_dict):
    # configurations initialization
    config = Config(model=model_name, dataset=dataset_name, config_dict=parameter_dict)

    # init random seed
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    # write config info into log
    logger.info(config)

    # print(config)
    # dataset creating and filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    return config, logger, dataset, train_data, valid_data, test_data


#################################################################################################################
#### Recbole Train and Evaluate functions

def get_trainer(model_type, model_name):
    r"""Copy of recbole.utils.utils modified to call CustomTrainer
    
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


def get_knn_method(df):
    print('number of users: ',df.user_id.nunique())
    print('number of items: ',df.item_id.nunique())
    if df.user_id.nunique() < df.item_id.nunique():
        print('Run User KNN')
        return 'user'
    else:
        print('Run Item KNN')
        return 'item'


def evaluate(trainer, test_data):
    test_result = None
    try:
        test_result = trainer.evaluate(test_data)
    except Exception as error:
        # handle the exception
        print("An exception occurred:", type(error).__name__)

        test_result = OrderedDict([(trainer.valid_metric, -1)])

    return test_result


def evaluate_print_error_message(trainer, test_data):
    test_result = None
    try:
        test_result = trainer.evaluate(test_data)
    except Exception as error:
        # handle the exception
        print("An exception occurred:", error)
        
        test_result = OrderedDict([(trainer.valid_metric, -1)])

    return test_result


def recbole_train_each_eval_all(model_name,
                                knn_method,
                                model_versions_to_train, 
                                model_versions_to_evaluate,
                                save_path, base_filename, specs_str,
                                filename_version,

                                use_gpu,
                                seed,
                                show_progress,
                                save_dataset,
                                shuffle,
                                benchmark_filename,
                                eval_args,
                                metrics,
                                ks,
                                valid_metric,
                            
                                part_shift_incl):

    model_class = None
    if model_name=='BPR':
        model_class = BPR
    elif model_name=='Pop':
        model_class = Pop
    elif model_name=='NeuMF':
        model_class = NeuMF
    elif model_name=='ItemKNN':
        model_class = ItemKNN
    else:
        raise Exception('Model name not expected! Current options are [BPR, Pop, NeuMF, ItemKNN]')  


    base_dataset_name = base_filename+'_'+specs_str

    for part in model_versions_to_train:
        print('\n\n'+part)
        
        # current data (ith part of the dataset) to feed the pre-trained model,
        # the result is referred to as "model part i" or "current model"
        current_dataset_name = base_dataset_name+part
        current_checkpoint_dir = save_path+current_dataset_name

        parameter_dict = {  'dataset': current_dataset_name+'.inter',
                            'use_gpu':use_gpu,

                            ## Environment settings https://recbole.io/docs/user_guide/config/environment_settings.html
                            'seed':seed,
                            'state':'INFO' if show_progress else 'ERROR', # ['INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL']
                            'data_path': save_path, # The path of input dataset.
                            'save_dataset':save_dataset, 
                            'checkpoint_dir':current_checkpoint_dir, # The path to save checkpoint file.
                            'show_progress': show_progress,
                            'shuffle': shuffle,

                            ## Data settings https://recbole.io/docs/user_guide/config/data_settings.html
                            'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
                            # 'user_inter_num_interval':'[1,inf)',
                            'benchmark_filename': benchmark_filename,
                            
                            ## Training settings https://recbole.io/docs/user_guide/config/training_settings.html
                            # 'train_neg_sample_args': TRAIN_NEG_SAMPLE_ARGS,
                            'knn_method':knn_method,

                            ## Evaluation settings https://recbole.io/docs/user_guide/config/evaluation_settings.html
                            'eval_args': eval_args,
                            'metrics': metrics, 
                            'topk':ks,
                            'valid_metric':valid_metric          
                        }


        # current_config,current_logger,current_dataset,current_train_data, current_valid_data, current_test_data
        current_config,\
            current_logger, _,\
                current_train_data,\
                    current_valid_data,\
                        current_test_data = setup_config_and_dataset(model_name,
                                                                    current_dataset_name,
                                                                    parameter_dict)

        # model loading and initialization
        current_model = model_class(current_config, current_train_data.dataset).to(current_config['device'])
        current_logger.info(current_model)


        # trainer loading and initialization
        trainer = get_trainer(current_config['MODEL_TYPE'], current_config['model'])(current_config, current_model)
        
        
        # model training
        best_valid_score, best_valid_result = trainer.fit(current_train_data, current_valid_data)
        print('\n\nTraining best results')
        print('best_valid_score: ', best_valid_score)
        print('best_valid_result: ', best_valid_result)


        # main diagonal eval
        test_result = evaluate(trainer, current_test_data)
        validate_and_save_picklefile(test_result,
                                    current_config['checkpoint_dir'], 
                                    get_evaluation_results_filename(current_config['model'], 
                                                                    current_dataset_name, 
                                                                    part, 
                                                                    filename_version))


        # evaluate in all testsets
        test_full_data_sections = get_test_full_data_sections_with_names(model_version=part,
                                                base_dataset_name=base_dataset_name,
                                                models_versions=model_versions_to_evaluate)
        
        test_full_data_sections = test_full_data_sections if part_shift_incl else test_full_data_sections[:-1]
        print(test_full_data_sections)

        for testset_name in test_full_data_sections:

            _,_,\
                _,trainset,_,\
                    testset = setup_config_and_dataset(model_name,
                                                        testset_name,
                                                        parameter_dict)

            # When calculate ItemCoverage metrics, we need to run this code for set item_nums in eval_collector.
            trainer.eval_collector.data_collect(trainset)
            
            # model evaluation
            test_result = evaluate(trainer, testset) # bc is the trainer that was just feed new data
            validate_and_save_picklefile(test_result,
                                        current_config['checkpoint_dir'],
                                        get_evaluation_results_filename(current_config['model'],
                                                                        current_dataset_name, 
                                                                        testset_name, 
                                                                        filename_version))    