from .filenames import *
from .file_utils import load_picklefile

import pandas as pd
from os import listdir
from os.path import isfile, join
from datetime import datetime


def interval_list_as_timestamp(split_intervals):
    '''
        example:
        split_intervals = [ ['2012-07','2012-12'],
                            ['2013-01','2013-06'],
                            ['2013-07','2013-12'],
                            ['2014-01','2014-06'],
                            ['2014-07','2015-01']
                        ]
    '''
    return [[pd.Timestamp(interval[0]), pd.Timestamp(interval[1])] for interval in split_intervals]



def get_user_recommendations_per_interval_df( model_versions_to_train,
                                              intervals_to_evaluate,
                                              algorithms,
                                              save_path, 
                                              base_filename, 
                                              specs_str,
                                              part_shift_incl):

    base_dataset_name = get_base_dataset_name(base_filename, specs_str)
    rec_foldername = 'recommendations'
    
    for part in model_versions_to_train:
        print('\n\nFetching recommended-lists files from model '+part)

        current_dataset_name = get_dataset_name(base_dataset_name, part)
        current_checkpoint_dir = get_checkpoint_dir(save_path, current_dataset_name) 
        print('\ncurrent_checkpoint_dir\n', current_checkpoint_dir)


        # get all recommendation file names
        rec_folderpath = get_rec_folderpath(current_checkpoint_dir, rec_foldername)
        print('\nrec_folderpath\n', rec_folderpath)
        filenames_df = pd.DataFrame([[  f,\
                                    f[f.index('_')+1:f.index('_test')],\
                                    f[f.rindex('_pt'):f.rindex('_batch')],\
                                    int(f[f.rindex('batch_')+6:f.rindex('_')]),\
                                    float(f[f.rindex('_')+1:f.rindex('.pkl')])]\
                                        for f in listdir(rec_folderpath) if isfile(join(rec_folderpath, f))],
                                columns=['name', 'algorithm', 'part', 'batch_id', 'time'])
        filenames_df.sort_values(by='time', inplace=True) # sort them by interval tested == time created



        
        for test_part in intervals_to_evaluate:
            print('\nEvaluation interval '+test_part)
            test_datasetname = get_dataset_name(base_dataset_name,test_part)
            test_checkpoint_dir = get_checkpoint_dir(save_path,test_datasetname)

            test_filepath = get_test_filepath(test_checkpoint_dir, test_datasetname)
            testset = pd.read_csv(test_filepath)

            
            print('Building the user profile (all the items viewed by the user @model '+test_part+')')
            train_filepath = get_train_filepath(test_checkpoint_dir, test_datasetname)
            trainset = pd.read_csv(train_filepath)
            # print(testset.shape, testset_reclists.shape, trainset.shape)
            _tt = trainset.loc[trainset['user_id'].isin(testset['user_id'].unique()), :]
            # print(testset.shape, testset_reclists.shape, trainset.shape, _tt.shape)
            # Group by user_id and get unique items as list
            user_profile = _tt.groupby('user_id')['item_id'].unique().reset_index()
            user_profile['profile_items'] = user_profile['item_id'].apply(list)
            user_profile.drop(columns=['item_id'], inplace=True)
            # print(testset.shape, testset_reclists.shape, trainset.shape, _tt.shape, user_profile.shape)  


            for algo in algorithms:
                print('\n\tRegarding the algorithm: ', algo)

                testset_reclists = pd.DataFrame()

                all_recfilenames = filenames_df.loc[(filenames_df.part==test_part) & (filenames_df.algorithm==algo),'name']
                for recfilename in all_recfilenames:                
                    rec_lists = load_picklefile(rec_folderpath+recfilename)
                    rec_lists = rec_lists.tolist() if hasattr(rec_lists, 'tolist') else rec_lists
                    n_reclists = len(rec_lists)

                    batch = testset[:n_reclists]   

                    testset_reclists = pd.concat( [ testset_reclists,\
                                                    pd.concat([batch, pd.DataFrame({'rec_list': rec_lists})], axis=1)],
                                                axis=0, 
                                                ignore_index=True)
                # print(testset_reclists.head())  
                # print(testset.shape, testset_reclists.shape)       
            

                user_rec_info_df = pd.merge(testset_reclists, user_profile[['user_id', 'profile_items']], on='user_id', how='left')
                # print('user_rec_info_df\n', user_rec_info_df.head())   
                # print(testset.shape, testset_reclists.shape, trainset.shape, _tt.shape, user_profile.shape, user_rec_info_df.shape)  


                testpt_to_interval = part_to_interval_map[test_part]
                pt_interval = split_intervals_dict[ testpt_to_interval ]
                user_rec_info_df['date'] = user_rec_info_df['timestamp'].apply(lambda x: datetime.fromtimestamp(x))
                

                user_rec_info_df['is_active_at_test_interval'] = 0
                user_rec_info_df.loc[ (user_rec_info_df.date >= pt_interval[0]) &\
                                    (user_rec_info_df.date <= pt_interval[1]),\
                                    'is_active_at_test_interval'] = 1
                user_rec_info_df['profile_size'] = user_rec_info_df['profile_items'].apply(lambda x : len(x))
                print('\tActive users @interval '+testpt_to_interval+'(if test inter. belongs to the interval, then the user was active)')


                user_rec_filepath = get_user_rec_filepath(current_checkpoint_dir,algo, test_part)
                user_rec_info_df.to_csv(user_rec_filepath, index=False)
                print('\tsaved '+user_rec_filepath)

                # print('user_rec_info_df\n', user_rec_info_df.head()) 
                # print(user_rec_info_df['is_active_at_test_interval'].value_counts())
                # print(user_rec_info_df['profile_size'].describe())
                # print(user_rec_info_df.loc[(user_rec_info_df['is_active_at_test_interval']==1)& (user_rec_info_df['profile_size'] > 20),:].shape)
                # print(user_rec_info_df.loc[(user_rec_info_df['is_active_at_test_interval']==1)& (user_rec_info_df['profile_size'] > 50),:].shape)
                # print(user_rec_info_df.loc[(user_rec_info_df['is_active_at_test_interval']==1)& (user_rec_info_df['profile_size'] > 80),:].shape)
                # print(user_rec_info_df.loc[(user_rec_info_df['is_active_at_test_interval']==1)& (user_rec_info_df['profile_size'] > 100),:].shape)
            
