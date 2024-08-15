import cv2
import numpy as np
import matplotlib.pyplot as plt

img_d = cv2.imread('/ibex/ai/home/liz0l/codes/datasets/cityscape/disparity/train/aachen/aachen_000008_000019_disparity.png', cv2.IMREAD_UNCHANGED).astype(np.float32)

print(img_d.shape)


mask = img_d == 0
plt.imshow(mask)
plt.savefig('./work_dir/mask.png')
