
"""

from PIL import Image
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

train_dir = r"/Users/eliyahunezri/Desktop/AI_applications/letters/original/"
lr = r"/Users/eliyahunezri/Desktop/AI_applications/letters/lr/"
hr = r"/Users/eliyahunezri/Desktop/AI_applications/letters/hr/"

# original_gray = r"/Users/eliyahunezri/Desktop/AI_applications/original_gray/"


for img in os.listdir(train_dir):

    if img != ".DS_Store":

        img_array = cv2.imread(train_dir + img)

        # contur_img = np.copy(img_array)
        # contur_img[:10, :] = [0, 0, 0]
        # contur_img[208:, :] = [0, 0, 0]
        # contur_img[:, :20] = [0, 0, 0]
        # contur_img[:, 1005:] = [0, 0, 0]
        # img_array = contur_img

        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        img_array = cv2.resize(img_array, (128, 128))
        lr_img_array = cv2.resize(img_array, (32, 32))

        cv2.imwrite(hr + img, img_array)
        cv2.imwrite(lr + img, lr_img_array)





# # for mix images
# path = r"/Users/eliyahunezri/Desktop/AI_applications/letters/hr/"
#
# hr_dist = r"/Users/eliyahunezri/Desktop/AI_applications/mix data/hr/"
# lr_dist = r"/Users/eliyahunezri/Desktop/AI_applications/mix data/lr/"
#
# # get the path/directory
#
# for images in os.listdir(path):
#
#
#     if images != ".DS_Store":
#
#         img = cv2.imread(path + "/" + images)
#
#         cv2.imwrite(hr_dist + images, img)

# name = 0
# path = r"/Users/eliyahunezri/Desktop/Downloads/dataset/test"
# for folder in os.listdir(path):
#     if folder != ".DS_Store":
#         for images in os.listdir(path + "/" + folder):
#             if images != ".DS_Store":
#                 img = cv2.imread(path + "/" + folder + "/" + images)
#
#                 cv2.imwrite(r"/Users/eliyahunezri/Desktop/AI_applications/letters/original/" + str(name) + ".png", img)
#                 name = name + 1
