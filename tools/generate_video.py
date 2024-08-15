import os
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image

images = '/ibex/ai/home/liz0l/projects/codes/PatchRefiner/data/kitti/raw/2011_09_26/2011_09_26_drive_0009_sync/image_02/data'
coarse = '/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/zoedepth/kitti/coarse_pretrain/vis_26_09'
pr = '/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/zoedepth/kitti/pr/vis_26_09'
ours = '/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/zoedepth/kitti/pr_sync_ssi_midas_grad_weight5/vis_26_09'


# path = images
# target_path = './work_dir/zoedepth/kitti_raw.mp4'
# resolution = (1216, 352)
# img_list = sorted(os.listdir(path))[200:]
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# video = cv2.VideoWriter(target_path, fourcc, 6, resolution)

# for i in tqdm(range(0, len(img_list))):
#     img = cv2.imread(os.path.join(path, img_list[i]))
    
#     height, width , _ = img.shape
#     top_margin = int(height - 352)
#     left_margin = int((width - 1216) / 2)
#     frame = img[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
#     video.write(frame)

# video.release()

########## coarse
# path = coarse
# target_path = './work_dir/zoedepth/kitti_coarse.mp4'
# resolution = (1216, 352)
# img_list = sorted(os.listdir(path))
# img_list = [img for img in img_list if '_edge' not in img and '_uint16' not in img and '.log' not in img][200:]
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# video = cv2.VideoWriter(target_path, fourcc, 6, resolution)

# for i in tqdm(range(0, len(img_list))):
    
#     frame = cv2.imread(os.path.join(path, img_list[i]))
#     frame = cv2.resize(frame, resolution, interpolation = cv2.INTER_LINEAR)
    
#     font = cv2.FONT_HERSHEY_COMPLEX_SMALL
#     cv2.putText(frame, 'ZoeDepth', (1050, 340), font, 1, color=(255, 255, 255), lineType=cv2.LINE_AA, thickness=2)
    
#     video.write(frame)

# video.release()

########## ours
# path = ours
# target_path = './work_dir/zoedepth/kitti_ours.mp4'
# resolution = (1216, 352)
# img_list = sorted(os.listdir(path))
# img_list = [img for img in img_list if '_edge' not in img and '_uint16' not in img and '.log' not in img][200:]
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# video = cv2.VideoWriter(target_path, fourcc, 6, resolution)

# for i in tqdm(range(0, len(img_list))):
    
#     frame = cv2.imread(os.path.join(path, img_list[i]))
#     frame = cv2.resize(frame, resolution, interpolation = cv2.INTER_LINEAR)
    
#     font = cv2.FONT_HERSHEY_COMPLEX_SMALL
#     # cv2.putText(frame, 'ZoeDepth', (50, 340), font, 1, color=(255, 255, 255), lineType=cv2.LINE_AA, thickness=2)
#     cv2.putText(frame, 'with S2R (Ours)', (20, 340-34), font, 1, color=(255, 255, 255), lineType=cv2.LINE_AA, thickness=2)
#     cv2.putText(frame, 'weight = 5', (55, 340-6), font, 1, color=(255, 255, 255), lineType=cv2.LINE_AA, thickness=2)
    
#     video.write(frame)

# video.release()

# ############# all
resolution = (1216, 352 * 3 + 12 * 2)
resolution_single = (1216, 352)

img_list = sorted(os.listdir(images))[200:]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('./work_dir/zoedepth/kitti.mp4', fourcc, 6, resolution)

coarse_list = [img for img in sorted(os.listdir(coarse)) if '_edge' not in img and '_uint16' not in img and '.log' not in img][200:]
pr_list = [img for img in sorted(os.listdir(pr)) if '_edge' not in img and '_uint16' not in img and '.log' not in img][200:]
ours_list = [img for img in sorted(os.listdir(ours)) if '_edge' not in img and '_uint16' not in img and '.log' not in img][200:]

for i in tqdm(range(0, len(img_list))):
    frame_1 = cv2.imread(os.path.join(images, img_list[i]))
    frame_2 = cv2.imread(os.path.join(coarse, img_list[i]))
    # frame_3 = cv2.imread(os.path.join(pr, img_list[i]))
    frame_4 = cv2.imread(os.path.join(ours, img_list[i]))
    
    height, width , _ = frame_1.shape
    top_margin = int(height - 352)
    left_margin = int((width - 1216) / 2)
    frame_1 = frame_1[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
    frame_2 = cv2.resize(frame_2, resolution_single, interpolation = cv2.INTER_LINEAR)
    # frame_3 = cv2.resize(frame_3, resolution_single, interpolation = cv2.INTER_LINEAR)
    frame_4 = cv2.resize(frame_4, resolution_single, interpolation = cv2.INTER_LINEAR)

    temp = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
    temp[0:352, :, :] = frame_1
    temp[352 + 12:352 * 2 + 12, :, :] = frame_2
    temp[352 * 2 + 12 * 2:352 * 3 + 12 * 2, :, :] = frame_4
    # temp[352 * 3 + 12 * 3:352 * 4 + 12 * 3, :, :] = frame_4
    
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    cv2.putText(temp, 'ZoeDepth', (1050, 352*2+12-12), font, 1, color=(255, 255, 255), lineType=cv2.LINE_AA, thickness=2)
    # cv2.putText(temp, 'w/o S2R', (1050, 352*3+12*2-12), font, 1, color=(255, 255, 255), lineType=cv2.LINE_AA, thickness=2)
    # cv2.putText(temp, 'with S2R (Ours)', (1000, 352*4+12*3-34), font, 1, color=(255, 255, 255), lineType=cv2.LINE_AA, thickness=2)
    # cv2.putText(temp, 'weight = 5', (1035, 352*4+12*3-6), font, 1, color=(255, 255, 255), lineType=cv2.LINE_AA, thickness=2)
    # cv2.putText(temp, 'with S2R (Ours)', (1000, 352*3+12*2-34), font, 1, color=(255, 255, 255), lineType=cv2.LINE_AA, thickness=2)
    cv2.putText(temp, 'Ours', (1075, 352*3+12*2-34), font, 1, color=(255, 255, 255), lineType=cv2.LINE_AA, thickness=2)
    cv2.putText(temp, 'weight = 5', (1035, 352*3+12*2-6), font, 1, color=(255, 255, 255), lineType=cv2.LINE_AA, thickness=2)
    
    video.write(temp)
    # cv2.imwrite('./work_dir/zoedepth/kitti/temp.jpg', temp)
    # exit(100)
video.release()
    