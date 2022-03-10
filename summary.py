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
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
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
parser.add_argument('--teacher_arch', type=str, default='densenet121', help='choose model type')
parser.add_argument('--model', type=str, default='densenet121', help='choose model type')
parser.add_argument('--student_arch', type=str, default='resnet50_stu', help='choose model type')

if __name__ == '__main__':

    args = parser.parse_args()
    args.auto_aug = False
    args.cutout = False
    args.auxiliary = True
    teacher_model = utils.get_network(args)

    genotype = eval('genotypes.%s' % args.arch)

    teacher_model.eval()
    # student_model = utils.get_student(args)
    # student_model.eval()

    # summary(teacher_model, input_size=(3, 256, 256), batch_size=1,device="cpu")
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    flops, params = profile(teacher_model, inputs=(torch.randn(1, 3, 256, 256),), verbose=False)
    print('flops = %fM,%',(flops / 1e6))
    print('param size = %fM', params / 1e6)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    stat(teacher_model,(3,256,256))