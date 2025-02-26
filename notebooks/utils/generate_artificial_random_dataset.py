import os
import pandas as pd
import random
from datetime import datetime
import time
import pickle

# import os
# import sys
# sys.path.append(os.path.abspath('') + '/..')


def validate_folderpath(folderpath):
    # print('im validating')
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
        print('Folder created: ', folderpath)


def save_picklefile(d, filepath):
    with open(filepath, 'wb') as f:
            pickle.dump(d, f)

    print('Saved file at '+filepath)

def load_picklefile(filepath):
    with open(filepath, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict

    

def create_folderpath(save_path, base_filename, specs_str):
    return save_path+base_filename+'_'+specs_str+'/'


def save_complete_dataset_atomic_file(df, save_path, base_filename, specs_str):
    if save_path:
        folderpath = create_folderpath(save_path, base_filename, specs_str)
        validate_folderpath(folderpath)
        # print(folderpath)
        # Output the dataset
        filepath = folderpath+base_filename+'_'+specs_str
        # print(filepath)
        df = df[['user_id', 'item_id', 'timestamp']]
        # print(df.head())

        df.to_csv(filepath+'.csv', index=False)
        df.to_csv(filepath+'.inter',
                            header=['user_id:token','item_id:token','timestamp:float'], 
                            sep='\t', 
                            index=False)
        print("Dataset with sudden drift created and saved at "+filepath+".")
        print(df.item_id.value_counts())


def generate_artificial_dataset(n_users,
                                n_items, 
                                ts,
                                all_items_seen,
                                random_seed,
                                n_items_to_drift,
                                sudden_drift_start,
                                drift_items_freq_list,
                                non_drift_items_freq_list,
                                save_path,
                                base_filename,
                                bin_size):
    

    def add_zero_user(df):

        df['user_id_n'] = df['user_id'].apply(lambda x: x[2:])
        df['user_id_n'] = df['user_id_n'].astype(int)

        df.loc[-1] = ['u_0', 'i_1', ts, 0]
        df.loc[-4] = ['u_0', 'i_5', ts, 0]
        df.loc[-6] = ['u_0', 'drifted_i_1', ts, 0]
        df.loc[-7] = ['u_0', 'drifted_i_5', ts, 0]

        df.sort_values(by='user_id_n', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.drop(columns=['user_id_n'], inplace=True)
        return df

    
    def create_folderpath(save_path, base_filename, specs_str):
        return save_path+base_filename+'_'+specs_str+'/'
    

    def save_items_frequencies(n_items_to_drift,
                               sudden_drift_start,
                               drift_items_freq_list,
                               non_drift_items_freq_list,
                               save_path, 
                               base_filename,
                               specs_str):
        
        folderpath = create_folderpath(save_path, base_filename, specs_str)
        validate_folderpath(folderpath)

        d = {'n_items_to_drift': n_items_to_drift,
             'sudden_drift_start': sudden_drift_start,
             'drift_items_freq_list': drift_items_freq_list,
             'non_drift_items_freq_list': non_drift_items_freq_list}

        save_picklefile(d, folderpath+'saved_dictionary.pkl')
       

    def save_complete_dataset_atomic_file(df, save_path, base_filename, specs_str):
        if save_path:
            folderpath = create_folderpath(save_path, base_filename, specs_str)
            validate_folderpath(folderpath)
            # Output the dataset
            filepath = folderpath+base_filename+'_'+specs_str

            df.to_csv(filepath+'.csv', index=False)
            df.to_csv(filepath+'.inter',
                                header=['user_id:token','item_id:token','timestamp:float'], 
                                sep='\t', 
                                index=False)
            print("Dataset with sudden drift created and saved at "+filepath+".")
            print(df.item_id.value_counts())



    def add_user0_save_dataset_sample_atomic_file(df,user_sample, save_path, base_filename, specs_str):
        if save_path:
            folderpath = create_folderpath(save_path, base_filename, specs_str)
            validate_folderpath(folderpath)
            # Output the dataset
            filepath = folderpath+base_filename+'_'+specs_str

            df_sampled = df[df.user_id.isin(user_sample)].reset_index(drop=True)
            df_sampled = add_zero_user(df_sampled)

            df_sampled.to_csv(filepath+'.csv', index=False)
            df_sampled.to_csv(filepath+'.inter',
                                header=['user_id:token','item_id:token','timestamp:float'], 
                                sep='\t', 
                                index=False)
            print("Dataset with sudden drift created and saved at "+filepath+".")
            print(df_sampled.item_id.value_counts())

    
    

    def add_all_users_save_dataset_sample_atomic_file(df,user_sample, save_path, base_filename, specs_str):
        user_sample = users_list[:bin_size]
        sampled_df = df[df.user_id.isin(user_sample)].reset_index(drop=True)
        # print('Number of user ids in the dataset TO BE part_1: ', sampled_df.user_id.nunique())

        sampled_df['user_id_n'] = sampled_df['user_id'].apply(lambda x: x[2:])
        sampled_df['user_id_n'] = sampled_df['user_id_n'].astype(int)

        # add all users, but user 0
        for i in range(1, n_users):
            sampled_df.loc[-i] = [users_list[i], 'i_1', ts, 0]
        sampled_df.reset_index(drop=True, inplace=True)

        # add user 0
        sampled_df.loc[-1] = ['u_0', 'i_1', ts, 0]
        sampled_df.loc[-2] = ['u_0', 'i_5', ts, 0]
        sampled_df.loc[-3] = ['u_0', 'drifted_i_1', ts, 0]
        sampled_df.loc[-4] = ['u_0', 'drifted_i_5', ts, 0]

        sampled_df.sort_values(by='user_id_n', inplace=True)
        sampled_df.reset_index(drop=True, inplace=True)
        sampled_df.drop(columns=['user_id_n'], inplace=True)

        print(sampled_df.head())
        print('Number of user ids in the dataset (all users alredy added): ', df.user_id.nunique())

        save_complete_dataset_atomic_file(sampled_df, save_path, base_filename, specs_str)


    def split_dataset_into_4_add_users_and_save_atomic_file(df, users_list, bin_size, save_path, base_filename, specs_str):
        '''
            add_all_users_save_dataset_sample_atomic_file()
                All users are added to the beginning of the dataset to prevent 'Index out of Range in self' Error, 
                    which is triggered by the presence user's ids in the test set of the future data 
                    and its absence in the train set of the model trained with the 1st section of the data, 
                    when the model is tested on future test sets, the error is yielded

            add_user0_save_dataset_sample_atomic_file()
                User0 sees i1, i5, i1_drifted, i5_drifted.
                    They are added to all sections to prevent the 'Some users have seen all items' Error,
                    which is triggered by the absence of i1, i5 in the 2nd half of the full dataset and the absence of i1_drifted, i5_drifted in the 1st half.            
        
        '''

        add_all_users_save_dataset_sample_atomic_file(df, users_list, save_path, base_filename, specs_str+'_with_all_users_at_beginning')
        add_user0_save_dataset_sample_atomic_file(df, users_list, save_path, base_filename, specs_str)

        add_all_users_save_dataset_sample_atomic_file(df, users_list[:bin_size], save_path, base_filename, specs_str+'_pt1')
        
        add_all_users_save_dataset_sample_atomic_file(df, users_list[:bin_size*2], save_path, base_filename, specs_str+'_pt2')
        add_all_users_save_dataset_sample_atomic_file(df, users_list[:bin_size*3], save_path, base_filename, specs_str+'_pt3')
                
        add_user0_save_dataset_sample_atomic_file(df, users_list[bin_size:bin_size*2], save_path, base_filename, specs_str+'_pt5')
        add_user0_save_dataset_sample_atomic_file(df, users_list[bin_size*2:bin_size*3], save_path, base_filename, specs_str+'_pt6')
        add_user0_save_dataset_sample_atomic_file(df, users_list[bin_size*3:], save_path, base_filename, specs_str+'_pt7')
        add_user0_save_dataset_sample_atomic_file(df, users_list[bin_size:bin_size*3], save_path, base_filename, specs_str+'_pt8')


    def calculate_sparsity(df):
        # df.item_id.groupby([df.user_id, df.item_id]).count().sum() == df.user_id.count()
        sparsity = 1 - df.user_id.count()/(df.user_id.nunique()*df.item_id.nunique())
        specs_str = str(df.user_id.nunique())+'x'+str(df.item_id.nunique())+'_'+str(round(sparsity, 2))
        print('specs_str', specs_str)
        return sparsity, specs_str
    

    def rename_item(row):
        if int(row['user_id'].split('_')[1]) > sudden_drift_start and row['item_id'] in renamed_items:
            return renamed_items[row['item_id']]
        return row['item_id']




    # users_list = [f'u_{i+1}' for i in range(1, n_users)]
    users_list = [f'u_{i+1}' for i in range(n_users)]
    items_list = [f'i_{j+1}' for j in range(n_items)]


    if all_items_seen:

        data = []
        for user in users_list:
            for item in items_list:
                data.append({'user_id': user, 'item_id': item, 'timestamp':ts})

        all_items_seen_df = pd.DataFrame(data)

        # Introduce sudden drift
        random.seed(random_seed)  # For reproducibility
        drift_items_list = random.sample(items_list, k=n_items_to_drift)  
        renamed_items = {item: f'drifted_{item}' for item in drift_items_list}       
        

        all_items_seen_df['item_id'] = all_items_seen_df.apply(rename_item, axis=1)
        # print(all_items_seen_df.item_id.groupby([all_items_seen_df.user_id, all_items_seen_df.item_id]).count().unstack().fillna(0).astype(int))

        
        sparsity , specs_str = calculate_sparsity(all_items_seen_df)
        print('sparsity: ',sparsity)

        sampled_df = add_zero_user(sampled_df)
        users_list.insert(0, 'u_0')
        
        split_dataset_into_4_add_users_and_save_atomic_file(sampled_df, users_list, bin_size, save_path, base_filename, specs_str)
        save_items_frequencies(n_items_to_drift, sudden_drift_start, drift_items_freq_list, non_drift_items_freq_list, save_path, base_filename, specs_str)
        

    else:
        
        if len(drift_items_freq_list) != n_items_to_drift:
            print('Not all items frequency was specified!')
            return None
        elif len(non_drift_items_freq_list) != len(items_list)-n_items_to_drift:
            print('Not all items frequency was specified!')
            return None


        def sample_with_repetition_of_pattern(users_list, items_list, items_freq_list):
            random.seed(random_seed)
            sampled_df = pd.DataFrame({})
            for i, freq in enumerate(items_freq_list):
                # print('k ',k)
                user_sample = random.sample(users_list[:sudden_drift_start], k=freq) +\
                                random.sample(users_list[sudden_drift_start:], k=freq)
                temp_df = pd.DataFrame({'user_id': user_sample, 
                                        'item_id': items_list[i]})
                # print(temp_df.item_id.groupby([temp_df.user_id, temp_df.item_id]).count().unstack().fillna(0).astype(int))
                sampled_df = pd.concat([sampled_df, temp_df])
            
            return sampled_df
        

        # Introduce sudden drift
        # No need to random sample, bc the list will have the frequencies for each item
        random.seed(random_seed)  # For reproducibility
        drift_items_list = random.sample(items_list, k=n_items_to_drift)  
        # drift_items_list = [items_list[i] for i,x in enumerate(items_freq_list) if x == sudden_drift_start]
        renamed_items = {item: f'drifted_{item}' for item in drift_items_list}
        non_drift_items_list = list(set(items_list) - set(drift_items_list))
        
        print('drift_items_list', drift_items_list)
        print('renamed_items', renamed_items)
        print('non_drift_items_list', non_drift_items_list)

        
        random.seed(random_seed)  # For reproducibility
        sampled_df = sample_with_repetition_of_pattern(users_list,
                                                       non_drift_items_list,
                                                       non_drift_items_freq_list)
        
        sampled_df = pd.concat([sampled_df,
                                sample_with_repetition_of_pattern(users_list,
                                                                    drift_items_list,
                                                                    drift_items_freq_list)])

        if sampled_df.user_id.nunique() < n_users:
            # print(sampled_df.head())
            users_not_sampled = list(set(users_list) - set(sampled_df.user_id))
            print('users_not_sampled', len(users_not_sampled))
            # print('drift_items_list', drift_items_list)
            for user in users_not_sampled:
                for item in drift_items_list:
                    # print(sampled_df.loc[sampled_df['user_id']==user, 'item_id'].count())
                    sampled_df.loc[len(sampled_df)] = [user, item]


        sampled_df['item_id'] = sampled_df.apply(rename_item, axis=1)
        # print(sampled_df.item_id.groupby([sampled_df.user_id, sampled_df.item_id]).count().unstack().fillna(0).astype(int))

        sampled_df['timestamp'] = ts


        sparsity, specs_str = calculate_sparsity(sampled_df)
        print('sparsity: ',sparsity)
        # print(specs_str)
        print(sampled_df.item_id.groupby([sampled_df.user_id, sampled_df.item_id]).count().unstack().fillna(0).astype(int))
        # print(sampled_df.head())


        # when trainning on pt1, yield ValueError: Some users have interacted with all items, which we can not sample negative items for them. Please set `user_inter_num_interval` to filter those users.
        # sampled_df = add_zero_user(sampled_df) # to solve the error 
        users_list.insert(0, 'u_0')

        # save_dataset_atomic_file(sampled_df, save_path, specs_str)
        split_dataset_into_4_add_users_and_save_atomic_file(sampled_df, users_list, bin_size, save_path, base_filename, specs_str)
        save_items_frequencies(n_items_to_drift, sudden_drift_start, drift_items_freq_list, non_drift_items_freq_list, save_path, base_filename, specs_str)


        return sampled_df



