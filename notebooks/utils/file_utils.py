
import os
import pickle

def validate_folderpath(folderpath):
    # print('im validating')
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
        print('Folder created: ', folderpath)


def save_picklefile(data, filepath):
    with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    print('Saved file at '+filepath)

def load_picklefile(filepath):
    with open(filepath, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict


def validate_and_save_picklefile(data, folderpath, filename):
    validate_folderpath(folderpath)
    save_picklefile(data, folderpath+'/'+filename+'.pkl')
