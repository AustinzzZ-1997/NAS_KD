import argparse
import glob
import logging
import os
import sys
import time
from signal import SIG_DFL, SIG_IGN, SIGPIPE, signal

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torchvision.datasets as dset

import utils
from architect import Architect
from model_search import Network
from separate_loss import ConvSeparateLoss, KD_loss, TriSeparateLoss

signal(SIGPIPE, SIG_IGN)

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.005, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--aux_loss_weight', type=float, default=10.0, help='weight decay')
parser.add_argument('--gpu', type=str, default='3', help='gpu device id, split with ","')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--single_level', action='store_true', default=False, help='use single level')
parser.add_argument('--sep_loss', type=str, default='l2', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--auto_aug', action='store_true', default=False, help='use auto augmentation')
parser.add_argument('--parse_method', type=str, default='threshold_sparse', help='parse the code method')
parser.add_argument('--op_threshold', type=float, default=0.7, help='threshold for edges')
parser.add_argument('--save', type=str, default='student_v1', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--arch_learning_rate', type=float, default=3e-3, help='learning rate for arch encoding')
parser.add_argument('--arch_lr_gamma', type=float, default=0.9, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--if_search', type=bool, default=True, help='if search choose dataset')
parser.add_argument('--dataset', type=str, default='cifar10', help='choose data')
parser.add_argument('--classes', type=int, default=10, help='classes')
parser.add_argument('--step', type=int,default=2, help='step number')
args = parser.parse_args()

args.save = './logs/search/'+'{}'.format(args.dataset) + '/{}-{}'.format(args.save, time.strftime('%Y%m%d-%H%M%S'))
# args.save = './logs/search_kd-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M"))
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
  # torch.cuda.set_device(args.gpu)
  gpus = [int(i) for i in args.gpu.split(',')]
  if len(gpus) == 1:
    torch.cuda.set_device(int(args.gpu))
  # cudnn.benchmark = True
  torch.manual_seed(args.seed)
  # cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %s' % args.gpu)
  logging.info("args = %s", args)

  criterion_train = ConvSeparateLoss(weight=args.aux_loss_weight) if args.sep_loss == 'l2' else TriSeparateLoss(weight=args.aux_loss_weight)

  criterion_val = nn.CrossEntropyLoss()

  model = Network(args.init_channels, args.classes, args.layers, criterion_train,
                  steps=args.step, multiplier=args.step, stem_multiplier=3,
                  parse_method=args.parse_method, op_threshold=args.op_threshold)
  model = model.cuda()



  if len(gpus)>1:
    print("True")
    model = nn.parallel.DataParallel(model, device_ids=gpus, output_device=gpus[0])
    model = model.module

  run_start = time.time()
  start_epoch = 0
  dur_time = 0
  best_acc = 0.0
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  model_optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  arch_optimizer = torch.optim.Adam(model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.9, 0.999), weight_decay=args.arch_weight_decay)

  train_queue,valid_queue = utils.get_dataloader(args)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        model_optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)
    # resume from checkpoint
  

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        model_optimizer, float(args.epochs), eta_min=args.learning_rate_min, last_epoch=-1 if start_epoch == 0 else start_epoch)

  for epoch in range(args.epochs):
    
    # lr = scheduler.get_lr()[0]
    lr = scheduler.get_last_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    print(F.softmax(model.alphas_normal, dim=-1))
    print(F.softmax(model.alphas_reduce, dim=-1))

    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion_train, model_optimizer, arch_optimizer)
    logging.info('train_acc %f', train_acc)

    # validation
   
    valid_acc, valid_obj = infer(valid_queue, model, criterion_val)
    logging.info('valid_acc %f', valid_acc)
    scheduler.step()
    if(valid_acc > best_acc):
      best_acc = valid_acc
      # utils.save(model, os.path.join(args.save, 'weights.pt'))
  logging.info('the best valid_acc %f', best_acc)
        # save checkpoint




def train(train_queue, valid_queue, model, architect, criterion, model_optimizer, arch_optimizer):
  objs = utils.AvgrageMeter()
  objs1 = utils.AvgrageMeter()
  objs2 = utils.AvgrageMeter()

  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    input = input.cuda()
    target = target.cuda()

    # Get a random minibatch from the search queue(validation set) with replacement
    # TODO: next is too slow
    if not args.single_level:
      input_search, target_search = next(iter(valid_queue))
      input_search = input_search.cuda()
      target_search = target_search.cuda()

    # bi-level default
    if not args.single_level:
      loss1, loss2 = architect.step(input_search, target_search)

    model_optimizer.zero_grad()

    ## if single-level
    if args.single_level:
      arch_optimizer.zero_grad()

    logits = model(input) #studentout
    aux_input = torch.cat([F.sigmoid(model.alphas_normal), F.sigmoid(model.alphas_reduce)], dim=0)

    # if not args.single_level:
    #   loss, _, _ = criterion(logits, target, aux_input)
    # else:
    #   loss, loss1, loss2 = criterion(logits, target, aux_input)

    loss, _ , _  = criterion(logits, target, aux_input)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    # Update the network parameters
    model_optimizer.step()

    ## if single level
    if args.single_level:
      arch_optimizer.step()


    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.item(), n)
    objs1.update(loss1, n)
    objs2.update(loss2, n)

    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
      logging.info('val cls_loss %e; spe_loss %e', objs1.avg, objs2.avg)
  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = input.cuda()
    target = target.cuda()

    logits = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main()

