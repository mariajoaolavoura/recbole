
def get_checkpoint_dir(save_path,dataset_name):
    return save_path+dataset_name

def get_dataset_name(base_dataset_name,part):
    return base_dataset_name+part

def get_base_dataset_name(base_filename, specs_str):
    return base_filename+'_'+specs_str

def get_rec_folderpath(checkpoint_dir, rec_foldername):
    return checkpoint_dir+'/'+rec_foldername+'/'

def get_test_filepath(checkpoint_dir, datasetname):
    return checkpoint_dir+'/'+datasetname+'.test'+'.csv'

def get_train_filepath(checkpoint_dir, datasetname):
    return checkpoint_dir+'/'+datasetname+'.train'+'.csv'

def get_user_rec_filepath(checkpoint_dir,algo, test_part, specs_str=''):
    return checkpoint_dir+'/'+algo+'_user_rec_info_test'+test_part+specs_str+'.csv'
            # ,\checkpoint_dir+'/'+algo+'_rec_ext_item_ids_test'+test_part+'.pkl'