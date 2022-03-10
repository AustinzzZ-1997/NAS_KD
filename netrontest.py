
import argparse

import netron
import torch
import torch.nn as nn
import torch.onnx
import torch.utils
import torchvision.datasets as dset
from thop import profile
from torch.autograd import Variable

import genotypes
import utils
from model import NetworkCIFAR as Network

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

genotype = eval('genotypes.%s' % args.arch)
model = Network(16, 10, 3, args.auxiliary, genotype, args.parse_method)
model.drop_path_prob = args.drop_path_prob
x = torch.rand(1, 3, 32, 32)
student_weight = './logs/train_kd/resnet50-cifar10/student_2step_v1-20211121-184706/weights.pt'

model.load_state_dict(torch.load(
      student_weight),strict=False)  # 仅加载参数
onnx_path = "../TVM/tiny_model.onnx"
input_names = ["input"]
output_names = ["output"]
torch.onnx.export(model, x, onnx_path,verbose=True,export_params=True,input_names=input_names, output_names=output_names)
# netron.start(onnx_path)
