
import os
from PIL import Image

image_path = ''
image = Image.open(image_path)
height = image.height
width = image.width
top_margin = int(height - 352)
left_margin = int((width - 1216) / 2)
image = image.crop(
    (left_margin, top_margin, left_margin + 1216, top_margin + 352))

output_path = os.path.join('./work_dir/kitti_images', image_path.replace('.jpg', '_crop.jpg').replace('/', '_'))
# image.save(image_path.replace('.jpg', '_crop.jpg'))