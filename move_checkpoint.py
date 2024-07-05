import os
checkpoint_folder = 'C:/Users/Administrator/Downloads/all_checkpoints'
# os.makedirs('./age_checkpoint', exist_ok=True)

# os.makedirs('./arcface_model', exist_ok=True)

# os.makedirs('./checkpoints', exist_ok=True)
# os.makedirs('./checkpoints/512', exist_ok=True)

# os.makedirs('./checkpoints/people', exist_ok=True)

# os.makedirs('./gan-control', exist_ok=True)
# os.makedirs('./gan-control/resources', exist_ok=True)
# os.makedirs('./gan-control/resources/gan_models', exist_ok=True)

# gan_models_path = './gan-control/resources/gan_models/'
# os.makedirs(gan_models_path+'controller_age015id025exp02hai04ori02gam15_temp', exist_ok=True)
# os.makedirs(gan_models_path+'controller_age015id025exp02hai04ori02gam15_temp/age_loss_20201013-061438', exist_ok=True)
# os.makedirs(gan_models_path+'controller_age015id025exp02hai04ori02gam15_temp/age_loss_20201013-061438/checkpoint', exist_ok=True)

# os.makedirs(gan_models_path+'controller_age015id025exp02hai04ori02gam15_temp', exist_ok=True)
# os.makedirs(gan_models_path+'controller_age015id025exp02hai04ori02gam15_temp/expression_loss_20201104-093349', exist_ok=True)
# os.makedirs(gan_models_path+'controller_age015id025exp02hai04ori02gam15_temp/expression_loss_20201104-093349/checkpoint', exist_ok=True)

# os.makedirs(gan_models_path+'controller_age015id025exp02hai04ori02gam15_temp', exist_ok=True)
# os.makedirs(gan_models_path+'controller_age015id025exp02hai04ori02gam15_temp/expression_q_loss_20201013-151038', exist_ok=True)
# os.makedirs(gan_models_path+'controller_age015id025exp02hai04ori02gam15_temp/expression_q_loss_20201013-151038/checkpoint', exist_ok=True)


# os.makedirs(gan_models_path+'controller_age015id025exp02hai04ori02gam15_temp', exist_ok=True)
# os.makedirs(gan_models_path+'controller_age015id025exp02hai04ori02gam15_temp/gamma_loss_20201104-093551', exist_ok=True)
# os.makedirs(gan_models_path+'controller_age015id025exp02hai04ori02gam15_temp/gamma_loss_20201104-093551/checkpoint', exist_ok=True)

# os.makedirs(gan_models_path+'controller_age015id025exp02hai04ori02gam15_temp', exist_ok=True)
# os.makedirs(gan_models_path+'controller_age015id025exp02hai04ori02gam15_temp/generator', exist_ok=True)
# os.makedirs(gan_models_path+'controller_age015id025exp02hai04ori02gam15_temp/generator/checkpoint', exist_ok=True)


# os.makedirs(gan_models_path+'controller_age015id025exp02hai04ori02gam15_temp', exist_ok=True)
# os.makedirs(gan_models_path+'controller_age015id025exp02hai04ori02gam15_temp/hair_loss_20201116-092804', exist_ok=True)
# os.makedirs(gan_models_path+'controller_age015id025exp02hai04ori02gam15_temp/hair_loss_20201116-092804/checkpoint', exist_ok=True)

# os.makedirs(gan_models_path+'controller_age015id025exp02hai04ori02gam15_temp', exist_ok=True)
# os.makedirs(gan_models_path+'controller_age015id025exp02hai04ori02gam15_temp/orientation_loss_20201013-094838', exist_ok=True)
# os.makedirs(gan_models_path+'controller_age015id025exp02hai04ori02gam15_temp/orientation_loss_20201013-094838/checkpoint', exist_ok=True)

# os.makedirs('./gan-control/src', exist_ok=True)
# os.makedirs('./gan-control/src/gan_control', exist_ok=True)
# os.makedirs('./gan-control/src/gan_control/projection', exist_ok=True)
# os.makedirs('./gan-control/src/gan_control/projection/lpips', exist_ok=True)
# os.makedirs('./gan-control/src/gan_control/projection/lpips/weights', exist_ok=True)
# os.makedirs('./gan-control/src/gan_control/projection/lpips/weights/v0.0', exist_ok=True)

# os.makedirs('./gan-control/src/gan_control/projection/lpips/weights/v0.1', exist_ok=True)

# os.makedirs('./gender-checkpoint', exist_ok=True)

# os.makedirs('./insightface_func', exist_ok=True)
# os.makedirs('./insightface_func/models', exist_ok=True)
# os.makedirs('./insightface_func/models/antelope', exist_ok=True)

# os.makedirs('./parsing_model', exist_ok=True)
# os.makedirs('./parsing_model/checkpoint', exist_ok=True)

# os.makedirs('./seg_swap', exist_ok=True)
# os.makedirs('./seg_swap/checkpoint', exist_ok=True)

def create_prefix_path(folder_tree_ls):
    prefix_path = ''
    if len(folder_tree_ls) >= 1: 
        for item in folder_tree_ls:
            prefix_path += item + '/'
    return prefix_path


def create_folder(folder_path):
    new_folder_paths = folder_path.split('+')
    for index, folder in enumerate(new_folder_paths):
        folder_tree_ls = new_folder_paths[:index]
        path = create_prefix_path(folder_tree_ls) + folder
        os.makedirs('./'+path, exist_ok=True)

import shutil
def copy_checkpoint(checkpoint_folder, old_folder_path):
    temp = old_folder_path
    old_folder_path = os.path.join(checkpoint_folder, old_folder_path)
    tmp = old_folder_path
    new_folder_path = temp.replace('+','/')
    for filename in os.listdir(tmp):
        print (tmp)
        checkpoint = os.path.join(tmp, filename)
        if not os.path.isfile(os.path.join(new_folder_path, os.path.basename(checkpoint))) :        
            shutil.copy(checkpoint, new_folder_path)
            print ('moving checkpoint ',checkpoint )
            print ('Done copy ',new_folder_path)
        else:
            print ('checkpoint', os.path.join(new_folder_path, os.path.basename(checkpoint)) , 'exist, move on')

for folder_path in os.listdir(checkpoint_folder):
    # create directory and move checkpoint 
    create_folder(folder_path)

for old_folder_path in os.listdir(checkpoint_folder):
    copy_checkpoint(checkpoint_folder, old_folder_path)