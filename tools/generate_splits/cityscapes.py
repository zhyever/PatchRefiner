
import os

# data_root = './data/cityscapes'
# split_val_file = './data/cityscapes/val.txt'

# with open(split_val_file, 'w') as f:

#     rgb_data_folder = os.path.join(data_root, 'leftImg8bit', 'val')
#     depth_data_folder = os.path.join(data_root, 'disparity', 'val')
    
#     rgb_file_dir = os.listdir(rgb_data_folder)
#     rgb_file_dir = sorted(rgb_file_dir)
    
#     for rgb_dir in rgb_file_dir:
#         rgb_file = os.listdir(os.path.join(rgb_data_folder, rgb_dir))
#         for idx, rgb_filename in enumerate(rgb_file):
#             rgb_file_path = os.path.join(rgb_data_folder, rgb_dir, rgb_filename)
#             depth_filename = rgb_filename.replace('leftImg8bit', 'disparity')
#             depth_file_path = os.path.join(depth_data_folder, rgb_dir, depth_filename)
            
#             if os.path.isfile(rgb_file_path) and os.path.isfile(depth_file_path):
#                 split_rgb = os.path.join('leftImg8bit', 'val', rgb_dir, rgb_filename)
#                 split_depth = os.path.join('disparity', 'val', rgb_dir, depth_filename)
#                 f.write(split_rgb + ' ' + split_depth + '\n')



data_root = './data/cityscapes'
split_train_file = './data/cityscapes/train.txt'

with open(split_train_file, 'w') as f:
    
    rgb_data_folder = os.path.join(data_root, 'leftImg8bit', 'train')
    depth_data_folder = os.path.join(data_root, 'disparity', 'train')
    
    rgb_file_dir = os.listdir(rgb_data_folder)
    rgb_file_dir = sorted(rgb_file_dir)
    
    for rgb_dir in rgb_file_dir:
        rgb_file = os.listdir(os.path.join(rgb_data_folder, rgb_dir))
        for idx, rgb_filename in enumerate(rgb_file):
            rgb_file_path = os.path.join(rgb_data_folder, rgb_dir, rgb_filename)
            depth_filename = rgb_filename.replace('leftImg8bit', 'disparity')
            depth_file_path = os.path.join(depth_data_folder, rgb_dir, depth_filename)
            
            if os.path.isfile(rgb_file_path) and os.path.isfile(depth_file_path):
                split_rgb = os.path.join('leftImg8bit', 'train', rgb_dir, rgb_filename)
                split_depth = os.path.join('disparity', 'train', rgb_dir, depth_filename)
                f.write(split_rgb + ' ' + split_depth + '\n')

    rgb_data_folder_extra = os.path.join(data_root, 'extra', 'leftImg8bit', 'train_extra')
    depth_data_folder_extra = os.path.join(data_root, 'extra', 'disparity', 'train_extra')

    rgb_file_dir = os.listdir(rgb_data_folder_extra)
    rgb_file_dir = sorted(rgb_file_dir)
    
    for rgb_dir in rgb_file_dir:
        if rgb_dir in ['augsburg', 'bad-honnef', 'bamberg']: # skip for visualization
            continue
        
        rgb_file = os.listdir(os.path.join(rgb_data_folder_extra, rgb_dir))
        for idx, rgb_filename in enumerate(rgb_file):
            rgb_file_path = os.path.join(rgb_data_folder_extra, rgb_dir, rgb_filename)
            depth_filename = rgb_filename.replace('leftImg8bit', 'disparity')
            depth_file_path = os.path.join(depth_data_folder_extra, rgb_dir, depth_filename)
            
            if os.path.isfile(rgb_file_path) and os.path.isfile(depth_file_path):
                split_rgb = os.path.join('extra', 'leftImg8bit', 'train_extra', rgb_dir, rgb_filename)
                split_depth = os.path.join('extra', 'disparity', 'train_extra', rgb_dir, depth_filename)
                f.write(split_rgb + ' ' + split_depth + '\n')