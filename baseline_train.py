import argparse
import glob
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
from torch.autograd import Variable
from torchvision import transforms

import genotypes
import utils
from model import NetworkCIFAR as Network
from utils import MyDataSet

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=2, help='gpu device id')
parser.add_argument('--epochs', type=int, default=70, help='num of training epochs')
parser.add_argument('--save', type=str, help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--model', type=str, default='resnet50', help='choose model type')
parser.add_argument('--classes', type=int, default=100, help='classes')
parser.add_argument('--dataset', type=str, default='cifar100', help='choose data')
parser.add_argument('--if_search', type=bool, default=False, help='if search choose dataset')
args = parser.parse_args()

args.save = './logs/teachertrain/'+'{}-{}'.format(args.model,args.dataset)+'/{}'.format(time.strftime("%Y%m%d-%H%M"))

utils.create_exp_dir(args.save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)



def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
   

    torch.cuda.set_device(int(args.gpu))
    # torch.cuda.set_device('cuda:2,3') #可指定多卡
    # cudnn.benchmark = True
    torch.manual_seed(args.seed)
    # cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %s' % args.gpu)
    logging.info("args = %s", args)

    best_acc = 0.0






    model = utils.get_network(args)
    # model = torch.nn.DataParallel(model)
# model = inceptionv3()
    model = model.cuda()

    
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
        )
    
    train_queue,valid_queue = utils.get_dataloader(args)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    
    for epoch in range(args.epochs):
  

        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        logging.info('train_acc %f', train_acc)

        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        logging.info('valid_acc %f', valid_acc)
        if(valid_acc > best_acc):
            best_acc = valid_acc
            utils.save(model, os.path.join(args.save, 'weights.pt'))
    logging.info('the best valid_acc %f', best_acc)


def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = input.cuda()
        target = target.cuda()

        optimizer.zero_grad()
        # print(input.shape) 8x3x450x450
        logits= model(input)

        loss = criterion(logits, target)

        loss.backward()
        
        optimizer.step()

        prec1, prec5= utils.accuracy(logits, target, topk=(1, 5))
        # print(prec1) #1.5625
        n = input.size(0)
        # print(n)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
     
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            
                    
    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = input.cuda()
    target = target.cuda()
    logits= model(input)
    loss = criterion(logits, target).cuda()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)
    if step % args.report_freq == 0:
            logging.info('val %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

