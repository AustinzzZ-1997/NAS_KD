import torch
import torchvision
import onnxruntime as rt
import numpy as np
import cv2
import argparse
import os
import sys
import time
import torch.utils
import torchvision.datasets as dset
from thop import profile
import utils
import genotypes
import utils as dutils
from model import NetworkCIFAR as Network
from torchsummary import summary

from torchstat import stat

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
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
train_transform, valid_transform = dutils._data_transforms_cifar(args)
valid_data = dset.CIFAR10(root=args.data, train=False, download=False, transform=valid_transform)
len_val = len(valid_data)
valid_queue = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)

genotype = eval('genotypes.%s' % args.arch)
#test image
img_path = "../TVM/cifar10_test/ariplane/airplane_img-1001.png"
img = cv2.imread(img_path)
img = cv2.resize(img, (32, 32))
img = np.transpose(img, (2, 0, 1)).astype(np.float32)
img = torch.from_numpy(img)
img = img.unsqueeze(0)

#pytorch test
# student_weight = './logs/train_kd/resnet50-cifar10/kd_2step_v2-20211118-192515/weights.pt'
# model = Network(16, 10, 8, args.auxiliary, genotype, args.parse_method)
# model.drop_path_prob = args.drop_path_prob
model = dutils.get_network(args)
weight = dutils.get_teacher_weight(args)
model.load_state_dict(torch.load(
    weight,map_location='cpu'),strict=False)  # 仅加载参数
model.eval()
output = model.forward(img)
val, cls = torch.max(output.data, 1)
print("[pytorch]--->predicted class:", cls.item())
print("[pytorch]--->predicted value:", val.item())

#onnx test
sess = rt.InferenceSession("../onnxmodel/densenet121.onnx")
x = "input"
y = ["out"]
output = sess.run(y, {x : img.numpy()})
cls = np.argmax(output[0][0], axis=0)
val = output[0][0][cls]
print("[onnx]--->predicted class:", cls)
print("[onnx]--->predicted value:", val)
