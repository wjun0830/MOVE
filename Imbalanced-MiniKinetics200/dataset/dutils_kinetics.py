# PATH_TO_LABELS
ROOT='./labels'

# FEATURE_NAME: PATH_TO_FEATURE

path_dict = {
        "ResNet101": '/project/2023_AAAI_MOVE/Imbalanced-MiniKinetics200/data/minikinetics_feature/ResNet101/',
        "ResNet50" : '/project/2023_AAAI_MOVE/Imbalanced-MiniKinetics200/data/minikinetics_feature/ResNet50/'}

# FEATURE_DIM
dim_dict = {
            "ResNet101": 2048,
            "ResNet50": 2048
            }

def get_feature_path(feature_name):
    return path_dict[feature_name] + 'train', path_dict[feature_name] + 'val'

def get_feature_dim(feature_name):
    return dim_dict[feature_name]

def get_label_path():
    lc_list = ROOT+'/count-labels-train.lst'
    train_list = ROOT+'/train.lst'
    val_list = ROOT+'/test.lst'
    return lc_list, train_list, val_list

def get_vallabel_path():
    valid_list = ROOT+'/validate.lst'
    return valid_list

