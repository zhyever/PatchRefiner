import os
import random

with open('/ibex/ai/home/liz0l/codes/datasets/scannet_pp_select_new/nvs_sem_train_raw.txt', 'r') as f:
    train_items = f.readlines()

valid_pairs = []
for item in train_items:
    image, depth = item.strip().split(' ')[0], item.strip().split(' ')[1]
    if os.path.isfile(image) and os.path.isfile(depth):
        valid_pairs.append("{} {}".format(image, depth))
        
print(len(valid_pairs))
# printing n elements from list
subset = random.sample(valid_pairs, 20000)

with open('/ibex/ai/home/liz0l/codes/datasets/scannet_pp_select_new/nvs_sem_train_subset.txt', 'w') as f:
    for item in subset:
        f.write("%s\n" % item)