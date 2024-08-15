import os
import cv2
images_dir = '/ibex/ai/home/liz0l/projects/codes/PatchRefiner/examples'

filenames = os.listdir(images_dir)

for image_name in filenames:
    image_path = os.path.join(images_dir, image_name)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (3840, 2160))
    cv2.imwrite(image_path.replace('.jpg', '_resize.jpg'), image)
