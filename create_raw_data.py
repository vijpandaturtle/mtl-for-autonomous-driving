# Creating a dataset of all images in a single folder

import glob
import os
from PIL import Image

root = "leftImg8bit_trainextra"
im_dir = root + "/data/all_images"
os.makedirs(im_dir)

# Images
train_im_list = glob.glob(root + "/leftImg8bit/train_extra/*")
counter = 0
for city in train_im_list:
    im_list = glob.glob(city + "/*.png")
    im_list.sort()
    for i in im_list:
        im = Image.open(i)
        im = im.resize((256, 128))
        im.save(im_dir + "/{}.png".format(counter))
        counter += 1
print("Training RGB images processing has completed.")
