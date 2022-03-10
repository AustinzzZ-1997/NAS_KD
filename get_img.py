# -*- coding:utf-8 -*-
# QQ2737499951奇迹
# 2019.12.13
 
 
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
 
 
 
classification = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
 
images_dir = "./test_img"
test_dir = "./test_img"
 
images_batch = "../data/cifar-10-batches-py/data_batch_"
test_batch = "../data/cifar-10-batches-py/test_batch"
 
 
def load_images(save_path, path_batch, num=1):
    data = []
    labels = []
    print("path_batch", path_batch)
    with open(path_batch, mode='rb') as file:
        data_dict = pickle.load(file, encoding='bytes')
        data += list(data_dict[b'data'])
        labels += list(data_dict[b'labels'])
    img = np.reshape(data, [-1, 3, 32, 32])
 
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for i in range(img.shape[0]):
        r = img[i][0]
        g = img[i][1]
        b = img[i][2]
 
        ir = Image.fromarray(r)
        ig = Image.fromarray(g)
        ib = Image.fromarray(b)
        rgb = Image.merge("RGB", (ir, ig, ib))
 
        if i < 10:  # 只显示前10张图片
            plt.imshow(rgb)
            plt.axis('off')  # 不显示坐标轴
            plt.show()
        label = classification[labels[i]]
        save_dir = save_path + label + "/"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        name = label + "_img-" + str(i) + ".png"
        rgb.save(save_dir + name, "PNG")  # 文件夹下是RGB融合后的图像
 
 
if __name__ == "__main__":
    # images
    print("正在保存images图片:")
    for i in range(5):
        path = images_batch + str(i + 1)
        load_images(images_dir, path)
    print("保存images完毕.")
    # test
    print("正在保存test图片:")
    load_images(test_dir, test_batch)
    print("保存test完毕.")