# PATH_TO_LABELS
ROOT='./labels'

# FEATURE_NAME: PATH_TO_FEATURE

path_dict = {
        "ResNet101": '/project/data/VideoLT/feats/ResNet101/ResNet101-feature',
        "TSM-R50"  : '/project/data/VideoLT/feats/TSM_R50/TSM-R50-feature',
        "ResNet-50" : '/project/data/VideoLT/feats/ResNet-50/ResNet-50-feature'}

# FEATURE_DIM
dim_dict = {
            "ResNet101": 2048,
            "TSM-R50": 2048,
            "ResNet50": 2048
            }

def get_feature_path(feature_name):
    return path_dict[feature_name]

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

