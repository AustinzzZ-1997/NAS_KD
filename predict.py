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

if __name__ == '__main__':

    args = parser.parse_args()
    args.auto_aug = False
    args.cutout = False
    args.auxiliary = True
    train_transform, valid_transform = dutils._data_transforms_cifar(args)
    # valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
    # len_val = len(valid_data)
    # valid_queue = torch.utils.data.DataLoader(valid_data, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)

    
    _,valid_queue = dutils.get_dataloader(args)
    teacher_model = utils.get_network(args)
    # teacher_model.cuda()
    teacher_weight = utils.get_teacher_weight(args)
  #   kd_save_path = './logs/train-resnet50_cifar100-20211106-1904/weights.pt'
    teacher_model.load_state_dict(torch.load(
      teacher_weight,map_location='cpu'),strict=False)  # 仅加载参数
    # teacher_model.cuda()
    # if not torch.cuda.is_available():
    #     print('no gpu device available')-
    #     sys.exit(1)


    genotype = eval('genotypes.%s' % args.arch)
    student_weight = './logs/train_kd/resnet50-cifar100/res50_stu_v2-20211123-140155/weights.pt'
    # student_weight = "./logs/train_kd/resnet50-cifar10/student_2step_v1-20211121-184706/weights.pt"
    model = Network(24, 100, 10, args.auxiliary, genotype, args.parse_method)
    model.drop_path_prob = args.drop_path_prob
    model.load_state_dict(torch.load(
    student_weight,map_location='cpu'),strict=False)  # 仅加载参数

    # student_weight = './logs/train_kd/resnet50-cifar10/student_2step_v1-20211121-184706/weights.pt'
    # student_weight = './logs/train_kd/resnet50-cifar10/kd_2step_v2-20211118-192515/weights.pt'
    # model = Network(16, 10, 8, args.auxiliary, genotype, args.parse_method)
    # model.drop_path_prob = args.drop_path_prob
    # model.load_state_dict(torch.load(
    #   student_weight,map_location='cpu'),strict=False)  # 仅加载参数
    # flops, params = profile(teacher_model, inputs=(torch.randn(1, 3, 32, 32),), verbose=False)
    # print('flops = %fM' % (flops / 1e6))
    # print('param size = %fM' %( params / 1e6))
    # summary(model, input_size=(3, 32, 32), batch_size=1,device="cpu")
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32),), verbose=False)
    # print('flops = %fM,%',(flops / 1e6))
    # print('param size = %fM', params / 1e6)
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # stat(model,(3,32,32))
    # model = model.cuda()

    # if args.model_path and os.path.isfile(args.model_path):
    #     checkpoint = torch.load(args.model_path)
    #     model.load_state_dict(checkpoint['state_dict'])
    # else:
    #     print('The Pre-Trained Model Is InValid!')
    #     sys.exit(-1)

    top1 = dutils.AvgrageMeter()
    top5 = dutils.AvgrageMeter()
    model.eval()
    teacher_model.eval()
    start = time.time()
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            # input = input.cuda(non_blocking=True)
            # target = target.cuda(non_blocking=True)
            logits,_ = model(input)
            # logits = teacher_model(input)
            prec1, prec5 = dutils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            # if step % args.report_freq == 0:
            #     print('valid %03d %f %f' % (step, top1.avg, top5.avg))
        print("Final Mean Top1: {}, Top5: {}".format(top1.avg, top5.avg))
    end = time.time()
    # print((end-start)/len_val)
        
