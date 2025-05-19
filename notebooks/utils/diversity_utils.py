from .popularity_inequality import PopularityInequality
from .train_evaluate import get_diversity_results_filename
from .file_utils import *
from .filenames import *
import pandas as pd
import numpy as np

from collections import  OrderedDict


def calculate_popularity_inequality(algorithms,
                                knn_method,
                                model_versions_to_train,
                                intervals_to_evaluate,
                                save_path,
                                base_dataset_name,
                                specs_str,
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
                                metric_decimal_place):


    # pt4_ds_name = get_dataset_name(base_dataset_name,'_pt4')
    # pt4_dir = get_checkpoint_dir(save_path,pt4_ds_name)
    # pt4_filepath = get_train_filepath(pt4_dir, pt4_ds_name)
    # pt4_df = pd.read_csv(pt4_filepath)
    # full_item_set_len = pt4_df.item_id.nunique()                                    

    for model_pt in model_versions_to_train:
        print('\n\nModel: ', model_pt)
        # model_pt = MODEL_VERSIONS[0] 
                                        
        dataset_name = get_dataset_name(base_dataset_name,model_pt)
        checkpoint_dir = get_checkpoint_dir(save_path,dataset_name)

        model_filepath = get_train_filepath(checkpoint_dir, dataset_name)
        model_df = pd.read_csv(model_filepath)
        # model_item_set_len = model_df.item_id.nunique()
        item_popularity_dict = model_df.item_id.value_counts().to_dict()



        for test_part in intervals_to_evaluate:
            for algo in algorithms:
                # test_part = MODEL_VERSIONS[0]
                # algo = ALGORITHMS[0]
                print('loading interval: ', test_part, '; algorithm: ', algo)


                user_rec_info_df = pd.read_csv(get_user_rec_filepath(checkpoint_dir, algo, test_part))
                active_users_df = user_rec_info_df.loc[user_rec_info_df['is_active_at_test_interval']==1, :].copy()


                user_profile_series = active_users_df.profile_items.apply(lambda x: eval(x))
                user_profile_item_ll = user_profile_series.to_numpy()
                # user_profile_matrix = np.vstack(np.asarray(user_profile_series, dtype=object))

                rec_item_series = active_users_df.rec_list.apply(lambda x: eval(x))
                print(len(rec_item_series))
                # rec_item_ll = rec_item_series.to_numpy()
                rec_item_matrix = np.vstack(np.asarray(rec_item_series, dtype=object))



                parameter_dict = {  'dataset': dataset_name+'.inter',
                                        'use_gpu':use_gpu,

                                        ## Environment settings https://recbole.io/docs/user_guide/config/environment_settings.html
                                        'seed':seed,
                                        'state':'INFO' if show_progress else 'ERROR', # ['INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL']
                                        'data_path': save_path, # The path of input dataset.
                                        'save_dataset':save_dataset, 
                                        'checkpoint_dir':checkpoint_dir, # The path to save checkpoint file.
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
                                        'valid_metric':valid_metric,

                                        'metric_decimal_place':metric_decimal_place
                                    }


                pi = PopularityInequality(parameter_dict)

                pi_rec = pi.calculate_metric_full_matrix({'rec.items':rec_item_matrix, 'data.item_popularity':item_popularity_dict})
                # print(pi_rec['popularityinequality'][:5], pi_rec['popularityinequality'].size)
                pi_profile = pi.calculate_metric_full_matrix({'rec.items':user_profile_item_ll, 'data.item_popularity':item_popularity_dict})
                # print(pi_profile['popularityinequality'][:5], pi_profile['popularityinequality'].size)
                
                active_users_df.loc[:, 'popularity_inequality@reclist'] = pi_rec['popularityinequality']
                active_users_df.loc[:, 'popularity_inequality@userprofile'] = pi_profile['popularityinequality']  
                active_users_df.loc[:, 'popularity_inequality_delta'] = abs(active_users_df.loc[:, 'popularity_inequality@userprofile'] - active_users_df.loc[:, 'popularity_inequality@reclist'])    
                active_users_df.to_csv(get_user_rec_filepath(checkpoint_dir, algo, test_part, specs_str='_au'), index=False)


                od = OrderedDict()
                od['popularity_inequality_delta'] = active_users_df['popularity_inequality_delta'].mean()
                validate_and_save_picklefile(od,
                                            checkpoint_dir,
                                            get_diversity_results_filename(algo,
                                                                            model_pt, 
                                                                            test_part,
                                                                            filename_version))  