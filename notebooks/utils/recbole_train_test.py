from .generate_artificial_random_dataset import save_picklefile, validate_folderpath
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR, Pop, ItemKNN, NeuMF
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger, get_trainer
from collections import OrderedDict

# import os
# import sys
# sys.path.append(os.path.abspath('') + '/..')

def get_evaluation_results_filename(model_name, model_part, section, filename_version=''):
    if section == 'diagonal' and model_part is None:
        return model_name+'_evaluation_results_diagonal'+filename_version
    elif model_part[-3:][:2] == 'pt': 
        return model_name+'_evaluation_results_model_'+model_part[-3:]+'_section_'+section[-3:]+filename_version
    else:
        return model_name+'_evaluation_results_model_full_section_'+section[-3:]+filename_version

def save_evaluation_results(test_Results,
                          folderpath,
                          filename):
    validate_folderpath(folderpath)
    save_picklefile(test_Results, folderpath+'/'+filename+'.pkl')


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


def train_evaluate(config, model, train_data, valid_data, test_data, parameter_dict, filename_version=''):
    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
    print('\n\nTraining best results')
    print('best_valid_score: ', best_valid_score)
    print('best_valid_result: ', best_valid_result)

    # model evaluation
    # test_result = trainer.evaluate(test_data)
    # model evaluation
    test_result = None
    try:
        test_result = trainer.evaluate(test_data)
    except Exception as error:
        # handle the exception
        print("An exception occurred:", type(error).__name__)
        test_result = OrderedDict([(trainer.valid_metric, -1)])
    
    print('\n\nTest results')
    print(test_result)
    save_evaluation_results(test_result, 
                            parameter_dict['checkpoint_dir'], 
                            get_evaluation_results_filename(config['model'], None, 'diagonal', filename_version))



def recbole_train_evaluate_bpr(model_name,
                           dataset_name,
                           parameter_dict,
                           filename_version=''):
    
    '''
        Example:

        K = 3
        VALID_METRIC = 'Recall@'+str(K)
        MODEL = 'BPR'
        SEED = 2020
        USE_GPU = False
        SHUFFLE = False 
        SHOW_PROGRESS = False

        base_filename = 'sudden_drift_dataset_all_users_start_i1i5_drift_all_parts'
        base_dataset_name = base_filename+'_4000x7_0.71'
        dataset_name=base_dataset_name+'_pt1'
        
        data_path = 'processed_datasets/artificial_data/'
        
        parameter_dict = {  'dataset': dataset_name+'.inter',
                            'data_path': data_path,
                            'load_col': {'inter': ['user_id', 'item_id']},
                            'use_gpu':USE_GPU,
                            'topk':K,
                            'valid_metric':VALID_METRIC,
                            'checkpoint_dir':data_path+dataset_name,
                            'seed':SEED,
                            'shuffle': SHUFFLE
                        }


        call: recbole_train_test_bpr(MODEL, dataset_name, parameter_dict)
    
    '''

    config,\
        logger,\
            dataset,\
                train_data,\
                    valid_data,\
                        test_data = setup_config_and_dataset(model_name,
                                                             dataset_name,
                                                             parameter_dict)


    # model loading and initialization
    model = BPR(config, train_data.dataset).to(config['device'])
    logger.info(model)

    train_evaluate(config, model, train_data, valid_data, test_data, parameter_dict, filename_version)
    

def recbole_train_evaluate_pop(model_name,
                           dataset_name,
                           parameter_dict,
                           filename_version=''):
    
    '''
        Example:

        K = 3
        VALID_METRIC = 'Recall@'+str(K)
        MODEL = 'BPR'
        SEED = 2020
        USE_GPU = False
        SHUFFLE = False 
        SHOW_PROGRESS = False

        base_filename = 'sudden_drift_dataset_all_users_start_i1i5_drift_all_parts'
        base_dataset_name = base_filename+'_4000x7_0.71'
        dataset_name=base_dataset_name+'_pt1'
        
        data_path = 'processed_datasets/artificial_data/'
        
        parameter_dict = { 

                        }

    
    '''

    config,\
        logger,\
            dataset,\
                train_data,\
                    valid_data,\
                        test_data = setup_config_and_dataset(model_name,
                                                             dataset_name,
                                                             parameter_dict)


    # model loading and initialization
    model = Pop(config, train_data.dataset).to(config['device'])
    logger.info(model)

    train_evaluate(config, model, train_data, valid_data, test_data, parameter_dict, filename_version)




def recbole_train_evaluate_itemknn(model_name,
                           dataset_name,
                           parameter_dict,
                           filename_version=''):
    
    '''
        Example:

        K = 3
        VALID_METRIC = 'Recall@'+str(K)
        MODEL = 'BPR'
        SEED = 2020
        USE_GPU = False
        SHUFFLE = False 
        SHOW_PROGRESS = False

        base_filename = 'sudden_drift_dataset_all_users_start_i1i5_drift_all_parts'
        base_dataset_name = base_filename+'_4000x7_0.71'
        dataset_name=base_dataset_name+'_pt1'
        
        data_path = 'processed_datasets/artificial_data/'
        
        parameter_dict = { 
        
                        }
    
    '''

    config,\
        logger,\
            dataset,\
                train_data,\
                    valid_data,\
                        test_data = setup_config_and_dataset(model_name,
                                                             dataset_name,
                                                             parameter_dict)


    # model loading and initialization
    model = ItemKNN(config, train_data.dataset).to(config['device'])
    logger.info(model)

    train_evaluate(config, model, train_data, valid_data, test_data, parameter_dict, filename_version)



def recbole_train_evaluate_neumf(model_name,
                           dataset_name,
                           parameter_dict,
                           filename_version=''):
    
    '''
        Example:

        K = 3
        VALID_METRIC = 'Recall@'+str(K)
        MODEL = 'BPR'
        SEED = 2020
        USE_GPU = False
        SHUFFLE = False 
        SHOW_PROGRESS = False

        base_filename = 'sudden_drift_dataset_all_users_start_i1i5_drift_all_parts'
        base_dataset_name = base_filename+'_4000x7_0.71'
        dataset_name=base_dataset_name+'_pt1'
        
        data_path = 'processed_datasets/artificial_data/'
        
        parameter_dict = { 
        
                        }
    
    '''

    config,\
        logger,\
            dataset,\
                train_data,\
                    valid_data,\
                        test_data = setup_config_and_dataset(model_name,
                                                             dataset_name,
                                                             parameter_dict)


    # model loading and initialization
    model = NeuMF(config, train_data.dataset).to(config['device'])
    logger.info(model)

    train_evaluate(config, model, train_data, valid_data, test_data, parameter_dict, filename_version)





def evaluate_on_data_sections(model_name,
                          current_model,
                          current_dataset_name,
                          current_train_data,
                          current_config,
                          model_checkpoint_file,
                          test_data_sections,
                          parameter_dict,
                          filename_version=''):


    # trainer loading and initialization
    trainer = get_trainer(current_config['MODEL_TYPE'], current_config['model'])(current_config, current_model)


    # results = []

    for datasec_name in test_data_sections:
        print('\n\n'+datasec_name)
        
        # earlier_config,\earlier_logger,\earlier_dataset,\earlier_train_data, earlier_valid_data, earlier_test_data 
        _,_,_,train_datasec,_, test_datasec = setup_config_and_dataset(model_name,
                                                            datasec_name,
                                                            parameter_dict)


        # When calculate ItemCoverage metrics, we need to run this code for set item_nums in eval_collector.
        trainer.eval_collector.data_collect(train_datasec)
        # trainer.eval_collector.data_collect(current_train_data)

        # model evaluation
        # test_result = trainer.evaluate(test_datasec, model_file=model_checkpoint_file)
        # results += [test_result]
        
        # model evaluation
        test_result = None
        try:
            test_result = trainer.evaluate(test_datasec, model_file=model_checkpoint_file)
        except Exception as error:
            # handle the exception
            print("An exception occurred:", type(error).__name__)
            test_result = OrderedDict([(trainer.valid_metric, -1)])
    
        print(test_result)
        # print('TODO: save file '+datasec_name)
        # print(current_dataset_name)
        save_evaluation_results(test_result, 
                          parameter_dict['checkpoint_dir'], 
                          get_evaluation_results_filename(model_name, current_dataset_name,datasec_name, filename_version)) #'test_results_model_'+current_dataset_name[-3:]+'_section_'+datasec_name[-3:])



def evaluate_on_data_sections_bpr(model_name,
                              model_checkpoint_file,
                              current_dataset_name,
                              test_data_sections,
                              parameter_dict,
                              filename_version=''):
        
    # current_config,current_logger,current_dataset,current_train_data, current_valid_data, current_test_data
    current_config,\
        current_logger, _,\
            current_train_data, _, _ = setup_config_and_dataset(model_name,
                                                                current_dataset_name,
                                                                parameter_dict)

    # model loading and initialization
    current_model = BPR(current_config, current_train_data.dataset).to(current_config['device'])
    current_logger.info(current_model)

    evaluate_on_data_sections(model_name=model_name,
                            current_model=current_model,
                            current_dataset_name=current_dataset_name,
                            current_train_data=current_train_data,
                            current_config=current_config,
                            model_checkpoint_file=model_checkpoint_file,
                            test_data_sections=test_data_sections,
                            parameter_dict=parameter_dict,
                            filename_version=filename_version)



def evaluate_on_data_sections_pop(model_name,
                              model_checkpoint_file,
                              current_dataset_name,
                              test_data_sections,
                              parameter_dict,
                              filename_version=''):
        
    # current_config,current_logger,current_dataset,current_train_data, current_valid_data, current_test_data
    current_config,\
        current_logger, _,\
            current_train_data, _, _ = setup_config_and_dataset(model_name,
                                                                current_dataset_name,
                                                                parameter_dict)

    # model loading and initialization
    current_model = Pop(current_config, current_train_data.dataset).to(current_config['device'])
    current_logger.info(current_model)

    evaluate_on_data_sections(model_name=model_name,
                            current_model=current_model,
                            current_dataset_name=current_dataset_name,
                            current_train_data=current_train_data,
                            current_config=current_config,
                            model_checkpoint_file=model_checkpoint_file,
                            test_data_sections=test_data_sections,
                            parameter_dict=parameter_dict,
                            filename_version=filename_version)



def evaluate_on_data_sections_itemknn(model_name,
                              model_checkpoint_file,
                              current_dataset_name,
                              test_data_sections,
                              parameter_dict,
                              filename_version=''):
        
    # current_config,current_logger,current_dataset,current_train_data, current_valid_data, current_test_data
    current_config,\
        current_logger, _,\
            current_train_data, _, _ = setup_config_and_dataset(model_name,
                                                                current_dataset_name,
                                                                parameter_dict)

    # model loading and initialization
    current_model = ItemKNN(current_config, current_train_data.dataset).to(current_config['device'])
    current_logger.info(current_model)

    evaluate_on_data_sections(model_name=model_name,
                            current_model=current_model,
                            current_dataset_name=current_dataset_name,
                            current_train_data=current_train_data,
                            current_config=current_config,
                            model_checkpoint_file=model_checkpoint_file,
                            test_data_sections=test_data_sections,
                            parameter_dict=parameter_dict,
                            filename_version=filename_version)
    


def evaluate_on_data_sections_neumf(model_name,
                              model_checkpoint_file,
                              current_dataset_name,
                              test_data_sections,
                              parameter_dict,
                              filename_version=''):
        
    # current_config,current_logger,current_dataset,current_train_data, current_valid_data, current_test_data
    current_config,\
        current_logger, _,\
            current_train_data, _, _ = setup_config_and_dataset(model_name,
                                                                current_dataset_name,
                                                                parameter_dict)

    # model loading and initialization
    current_model = NeuMF(current_config, current_train_data.dataset).to(current_config['device'])
    current_logger.info(current_model)

    evaluate_on_data_sections(model_name=model_name,
                            current_model=current_model,
                            current_dataset_name=current_dataset_name,
                            current_train_data=current_train_data,
                            current_config=current_config,
                            model_checkpoint_file=model_checkpoint_file,
                            test_data_sections=test_data_sections,
                            parameter_dict=parameter_dict,
                            filename_version=filename_version)



def get_test_three_main_diagonal_sections_with_names(model_version:str, base_dataset_name:str):
    """
        model_version:str ['_pt1', '_pt2', '_pt3', '']

            Dataset section parts
                |    part 8    |
        |part 1|part 5| part 6|part 7|
        |    part 2   |
        |         part 3      |
        |             ''             |
    """
    test_datasec = []
    if model_version=='_pt1':
        test_datasec = [base_dataset_name+'_pt5']
    elif model_version=='_pt2':
        test_datasec = [base_dataset_name+'_pt1', base_dataset_name+'_pt6']
    elif model_version=='_pt3':
        test_datasec = [base_dataset_name+'_pt5', base_dataset_name+'_pt7']
    else:
        # model_version==''or '_pt4'
        test_datasec = [base_dataset_name+'_pt6']
    return test_datasec


def get_test_data_sections(model_version:str, 
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
    
    duration = len(models_versions)
    n_parts = duration*2+1
    triangle_holdouts = [models_versions[0]]+['_pt'+str(i) for i in range(duration+1, n_parts)]

    for i, pt in enumerate(models_versions):
        if model_version==pt:
            return triangle_holdouts[:i]+triangle_holdouts[i+1:]
    

def get_test_data_sections_with_names(model_version:str, 
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
    test_datasec = get_test_data_sections(model_version, models_versions)

    return [base_dataset_name+datasec for datasec in test_datasec]


# def get_test_data_sections_with_names(model_version:str, base_dataset_name:str):
#     """
#         model_version:str ['_pt1', '_pt2', '_pt3', '']

#             Dataset section parts
#                 |    part 8    |
#         |part 1|part 5| part 6|part 7|
#         |    part 2   |
#         |         part 3      |
#         |             ''             |
#     """
#     test_datasec = []
#     if model_version=='_pt1':
#         test_datasec = [base_dataset_name+'_pt5', base_dataset_name+'_pt6', base_dataset_name+'_pt7', base_dataset_name+'_pt8']
#     elif model_version=='_pt2' or model_version=='_pt5':
#         test_datasec = [base_dataset_name+'_pt1', base_dataset_name+'_pt6', base_dataset_name+'_pt7', base_dataset_name+'_pt8']
#     elif model_version=='_pt3' or model_version=='_pt6':
#         test_datasec = [base_dataset_name+'_pt1', base_dataset_name+'_pt5', base_dataset_name+'_pt7', base_dataset_name+'_pt8']
#     else:
#         # model_version==''or '_pt4' or '_pt7'
#         test_datasec = [base_dataset_name+'_pt1', base_dataset_name+'_pt5', base_dataset_name+'_pt6', base_dataset_name+'_pt8']
#     return test_datasec



# def get_test_data_sections(model_version:str):
#     """
#         model_version:str ['_pt1', '_pt2', '_pt3', '']

#             Dataset section parts
#                 |    part 8    |
#         |part 1|part 5| part 6|part 7|
#         |    part 2   |
#         |         part 3      |
#         |             ''             |
#     """
#     test_datasec = []
#     if model_version=='_pt1':
#         test_datasec = ['_pt5', '_pt6', '_pt7', '_pt8']
#     elif model_version=='_pt2' or model_version=='_pt5':
#         test_datasec = ['_pt1', '_pt6', '_pt7', '_pt8']
#     elif model_version=='_pt3' or model_version=='_pt6':
#         test_datasec = ['_pt1', '_pt5', '_pt7', '_pt8']
#     else:
#         # model_version=='' or '_pt4'
#         test_datasec = ['_pt1', '_pt5', '_pt6', '_pt8']
#     return test_datasec