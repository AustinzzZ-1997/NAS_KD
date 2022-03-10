import cv2
import time
import torch
import numpy as np
from torch.nn import DataParallel

from collections import OrderedDict
from torchvision import transforms as T
import onnxruntime as rt
 
import argparse
import os
import sys
import time
import torch.utils
import torchvision.datasets as dset
from thop import profile

import utils as dutils
from torchsummary import summary

from torchstat import stat

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--arch', type=str, default='studentarch_2step_v1', help='which architecture to use')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--parse_method', type=str, default='threshold', help='experiment name')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--if_search', type=bool, default=False, help='if search choose dataset')
parser.add_argument('--classes', type=int, default=10, help='classes')
parser.add_argument('--alpha', type=float,default=0.8, help='teacherout coefficient')
parser.add_argument('--T', type=float, default=10.0, help='soft label temprature')
parser.add_argument('--step', type=int,default=2, help='step number')
parser.add_argument('--dataset', type=str, default='cifar10', help='choose data')
parser.add_argument('--teacher_arch', type=str, default='resnet50', help='choose model type')
parser.add_argument('--model', type=str, default='resnet50', help='choose model type')



args = parser.parse_args()
args.auto_aug = False
args.cutout = False
args.auxiliary = True





_,valid_queue = dutils.get_dataloader(args)    

 
# def img_process(img_path):
#     normalize = T.Normalize(mean = [0.5, 0.5, 0.5],
#                             std = [0.5, 0.5, 0.5])
#     transforms = T.Compose([T.ToTensor(),
#                             normalize])
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (32, 32))
#     img = transforms(img)
#     img = img.unsqueeze(0)
#     return img
 
 
if __name__ == "__main__":
    sess = rt.InferenceSession("../onnxmodel/res50_0.18M_94.55.onnx")
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    
    flag = 0.0
    t0 = time.time()
    for step, (input, target) in enumerate(valid_queue):
        # img = 
        output = sess.run([output_name], {input_name:np.array(input)})
        # print(pred_onnx)
        cls = np.argmax(output[0][0], axis=0) #输出的是类
        if cls == target:
            flag = flag +1
            # if step % args.report_freq == 0:
            #     print('valid %03d %f %f' % (step, top1.avg, top5.avg))
    print("acc: {}".format(flag/len(valid_queue)))
    t1 = time.time()
