import os
import random
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

import genotypes
from auto_augment import CIFAR10Policy


class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].reshape(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124] if args.dataset == 'cifar10' else [0.50707519, 0.48654887, 0.44091785]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768] if args.dataset == 'cifar10' else [0.26733428, 0.25643846, 0.27615049]


  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])

  # if args.auto_aug:
  #   random_transform += [CIFAR10Policy()]

  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform

def _data_transforms_imagenet(args):
  CIFAR_MEAN = [0.485, 0.456, 0.406]
  CIFAR_STD = [0.229, 0.224, 0.225]

  train_transform = transforms.Compose([
    # transforms.RandomCrop(64),
    transforms.Resize([64,64]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])


  valid_transform = transforms.Compose([
    transforms.Resize([64,64]),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform

def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  # state_dict = torch.load(model_path)
  # # create new ordererDict that does not contain 'module'
  # from collections import OrderedDict
  # new_state_dict = OrderedDict()
  # for k, v in state_dict.items():
  #   namekey = k[7:] # remove 'module'
  #   new_state_dict[namekey] = v
  # # load params
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
    # mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path):
  if not os.path.exists(path):
    os.makedirs(path)
  print('Experiment dir : {}'.format(path))



def _data_transforms_cifar(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124] if args.dataset == 'cifar10' else [0.50707519, 0.48654887, 0.44091785]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768] if args.dataset == 'cifar10' else [0.26733428, 0.25643846, 0.27615049]

  normalize_transform = [
      transforms.ToTensor(),
      transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]

  random_transform = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip()]

  if args.auto_aug:
    random_transform += [CIFAR10Policy()]

  if args.cutout:
    cutout_transform = [Cutout(args.cutout_length)]
  else:
    cutout_transform = []

  train_transform = transforms.Compose(
      random_transform + normalize_transform + cutout_transform
  )

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform

def get_network(args):
    """ return given network
    """

    if args.model == 'resnet50':
        from teachermodel.resnet import resnet50
        return resnet50(num_classes=args.classes)
    elif args.model == 'resnet152':
        from teachermodel.resnet import resnet152
        return resnet152(num_classes=args.classes)    
    elif args.model == 'vgg16':
        from teachermodel.vgg import vgg16_bn
        return vgg16_bn(args.classes)
    elif args.model == 'vgg19':
        from teachermodel.vgg import vgg19_bn
        return vgg19_bn(args.classes)
    elif args.model == 'densenet121':
        from teachermodel.densenet import densenet121
        return densenet121(num_classes=args.classes)
    elif args.model == 'densenet201':
        from teachermodel.densenet import densenet201
        return densenet201(num_class=args.classes)
    elif args.model == 'inceptionv3':
        from teachermodel.inceptionnet import inceptionv3

        # from torchvision.models.inception import inception_v3
        return inceptionv3(args.classes)
    elif args.model == 'efficientnet_b4':
        from teachermodel.efficentnet import efficientnet_b4

        # from torchvision.models.inception import inception_v3
        return efficientnet_b4(args.classes)


def get_teacher_weight(args):
    """ return given network
    """

    if args.teacher_arch == 'resnet50':
        if args.dataset == 'cifar10':
          path = './logs/teachertrain/train-resnet50_cifar10-20211115-1708/weights.pt' #96.32%
        elif args.dataset == 'cifar100':
          path = './logs/teachertrain/train-resnet50_cifar100-20211114-2118/weights.pt' #78.29%
        elif args.dataset == 'vegetables':
          path = './logs/teachertrain/resnet50-vegetables/20211217-1831/weights.pt' #95.52%
          # path = './logs/teachertrain/resnet50-vegetables/20211214-2024/weights.pt'

    elif args.teacher_arch == 'resnet152':
        if args.dataset == 'cifar10':
          path = './logs/teachertrain/resnet152-cifar10/20211119-1415/weights.pt' #96.45%
        elif args.dataset == 'cifar100':
          path = './logs/teachertrain/resnet152-vegetables/20211217-1833/weights.pt' #70.32%
        elif args.dataset == 'vegetables':
          path = './logs/teachertrain/resnet152-vegetables/20211217-1959/weights.pt' #95.52%

    elif args.teacher_arch == 'vgg19':
        if args.dataset == 'cifar10':
          path = './logs/teachertrain/train-vgg19_cifar10-20211115-1958/weights.pt' #95.02%
        elif args.dataset == 'cifar100':
          path = './logs/teachertrain/train-vgg19_cifar100-20211114-2048/weights.pt' #74.31%
        elif args.dataset == 'vegetables':
          path = './logs/teachertrain/vgg19-vegetables/20211217-2024/weights.pt' #95.52%
    elif args.teacher_arch == 'vgg16':
        if args.dataset == 'cifar10':
          path = './logs/teachertrain/vgg16-cifar10/20211215-1603/weights.pt' #95.02%
        elif args.dataset == 'cifar100':
          path = './logs/teachertrain/vgg16-cifar100/20211215-2023/weights.pt' #74.31%
        # elif args.dataset == 'vegetables':
        #   path = './logs/train-resnet50_shucai-20211114-1727/weights.pt' #95.52%
    elif args.teacher_arch == 'densenet121':
        if args.dataset == 'cifar10':
          path = './logs/teachertrain/train-densenet121_cifar10-20211115-1959/weights.pt' #96.53%
        elif args.dataset == 'cifar100':
          path = './logs/teachertrain/train-densenet121_cifar100-20211115-1704/weights.pt' #79.41%
        elif args.dataset == 'vegetables':
          path = './logs/teachertrain/densenet201-vegetables/20211217-1834/weights.pt' #97.05%
    # elif args.model == 'densenet201':
    #     from teachermodel.densenet import densenet201
    #     return densenet201(args.classes)
    elif args.teacher_arch == 'densenet201':
        if args.dataset == 'cifar10':
          path = './logs/teachertrain/densenet201-cifar10/20211119-1415/weights.pt' #94.8%
        elif args.dataset == 'cifar100':
          path = './logs/teachertrain/densenet201-cifar100/20211228-1314/weights.pt' #70.32%
        elif args.dataset == 'vegetables':
          path = './logs/teachertrain/densenet201-vegetables/20211217-1834/weights.pt' #97.05%

    elif args.teacher_arch == 'inceptionv3':
        if args.dataset == 'cifar10':
          path = './logs/teachertrain/train-inceptionv3_cifar10-20211115-2000/weights.pt' #96.45%
        elif args.dataset == 'cifar100':
          path = './logs/teachertrain/train-inceptionv3_cifar100-20211114-2213/weights.pt' #70.32%
    elif args.model == 'efficientnet_b4':
        if args.dataset == 'vegetables':
          path = './logs/teachertrain/train-EFb4_shucai-20211114-1830/weights.pt' #89.22%
        else : 
          pass
    return path

def get_student(args):
  from model import NetworkImageNet as Network
  if args.student_arch == 'resnet50_stu':
        if args.dataset == 'cifar10':
          args.arch = 'studentarch_2step_v1'
          genotype = eval('genotypes.%s' % args.arch)
          student_weight = './logs/train_kd/resnet50-cifar10/kd_2step_v2-20211118-192515/weights.pt' 
          layer = 8
          channel = 16
        elif args.dataset == 'cifar100':
          args.arch = 'resnet50_cifar100_2step'
          genotype = eval('genotypes.%s' % args.arch)
          student_weight = './logs/train_kd/resnet50-cifar100/res50_stu_v2-20211123-140155/weights.pt' 
          layer = 10
          channel = 24
        elif args.dataset == 'vegetables':
          args.arch = 'resnet50_vege_stu'
          genotype = eval('genotypes.%s' % args.arch)
          student_weight = './logs/train_kd/resnet50-vegetables/stu_v1-20211222-155507/weights.pt' 
          layer = 10
          channel = 16
  if args.student_arch == 'resnet152_stu':
        if args.dataset == 'cifar10':
          args.arch = 'resnet152_stu_2step_v2'
          genotype = eval('genotypes.%s' % args.arch)
          student_weight = './logs/train_kd/resnet152-cifar10/stu_v2-20211206-164348/weights.pt' 
          layer = 8
          channel = 16
        elif args.dataset == 'cifar100':
          args.arch = 'resnet152_cifar100_2step'
          genotype = eval('genotypes.%s' % args.arch)
          student_weight = './logs/train_kd/resnet152-cifar100/stu_v1-20211128-180333/weights.pt' 
          layer = 8
          channel = 24
        elif args.dataset == 'vegetables':
          args.arch = 'resnet152_vege_stu'
          genotype = eval('genotypes.%s' % args.arch)
          student_weight = './logs/train_kd/resnet152-vegetables/stu_v1-20211226-190920/weights.pt' 
          layer = 10
          channel = 16
  if args.student_arch == 'vgg19_stu':
        if args.dataset == 'cifar10':
          args.arch = 'vgg19_stu_2step'
          genotype = eval('genotypes.%s' % args.arch)
          student_weight = './logs/train_kd/vgg19-cifar10/vgg19_stu_v1-20211123-140620/weights.pt' 
          layer = 8
          channel = 16
        elif args.dataset == 'cifar100':
          args.arch = 'vgg19_cifar100_2step'
          genotype = eval('genotypes.%s' % args.arch)
          student_weight = './logs/train_kd/vgg19-cifar100/stu_v2-20211208-150712/weights.pt' 
          layer = 12
          channel = 24
        elif args.dataset == 'vegetables':
          os.error("no path")
  if args.student_arch == 'vgg16_stu':
        if args.dataset == 'cifar10':
          args.arch = 'vgg16_cifar10_stu'
          genotype = eval('genotypes.%s' % args.arch)
          student_weight = './logs/train_kd/vgg16-cifar10/stu_v1-20211220-170452/weights.pt' 
          layer = 8
          channel = 16
        elif args.dataset == 'cifar100':
          args.arch = 'vgg16_cifar100_stu'
          genotype = eval('genotypes.%s' % args.arch)
          student_weight = './logs/train_kd/vgg16-cifar100/stu_v1-20211220-170502/weights.pt' 
          layer = 10
          channel = 16
        elif args.dataset == 'vegetables':
          os.error("no path")
  if args.student_arch == 'densenet201_stu':
        if args.dataset == 'cifar10':
          args.arch = 'densenet201_stu_2step'
          genotype = eval('genotypes.%s' % args.arch)
          student_weight = './logs/train_kd/densenet201-cifar10/stu_v1-20211127-142021/weights.pt' 
          layer = 8
          channel = 24
        elif args.dataset == 'cifar100':
          args.arch = 'densenet201_cifar100_stu'
          genotype = eval('genotypes.%s' % args.arch)
          student_weight = './logs/train_kd/densenet201-cifar100/stu_v1-20220107-180137/weights.pt' 
          layer = 10
          channel = 24
        elif args.dataset == 'vegetables':
          args.arch = 'densenet201_vege_stu'
          genotype = eval('genotypes.%s' % args.arch)
          student_weight = './logs/train_kd/densenet201-vegetables/stu_v1-20211226-190924/weights.pt' 
          layer = 10
          channel = 16


       
  if args.student_arch == 'inceptionv3_stu':
        if args.dataset == 'cifar10':
          args.arch = 'inceptionv3_stu_v1'
          genotype = eval('genotypes.%s' % args.arch)
          student_weight = './logs/train_kd/inceptionv3-cifar10/stu_v1-20211128-181032/weights.pt' 
          layer = 8
          channel = 24
        elif args.dataset == 'cifar100':
          args.arch = 'inceptionv3_cifar100_2step'
          genotype = eval('genotypes.%s' % args.arch)
          student_weight = './logs/train_kd/inceptionv3-cifar100/stu_v2-20211208-150510/weights.pt' 
          layer = 12
          channel = 24
        elif args.dataset == 'vegetables':
          os.error("no path")
    
  model = Network(channel, args.classes, layer, args.auxiliary, genotype, args.parse_method)
  model.drop_path_prob = args.drop_path_prob
  model.load_state_dict(torch.load(
    student_weight,map_location='cpu'),strict=False)  # 仅加载参数
  return model

def get_dataloader(args):
    if args.dataset == 'cifar10':
      train_transform, valid_transform = _data_transforms_cifar10(args)
      train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
      valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
      if args.if_search == True:
        num_train = len(train_data) #50000
        indices = list(range(num_train))
        split = int(np.floor(args.train_portion * num_train))

        train_queue = DataLoader(
        train_data, batch_size=args.batch_size, sampler=SubsetRandomSampler(indices[:split]),pin_memory=True, num_workers=5)

        valid_queue = DataLoader(
        train_data, batch_size=args.batch_size, sampler=SubsetRandomSampler(
              indices[split:num_train]),pin_memory=True, num_workers=5)
      else:
        train_queue = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=1)

        valid_queue = DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=1)
    elif args.dataset == 'cifar100':
      train_transform, valid_transform = _data_transforms_cifar10(args)
      train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
      valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
      if args.if_search == True:
        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(args.train_portion * num_train))
        train_queue = DataLoader(
        train_data, batch_size=args.batch_size, sampler=SubsetRandomSampler(indices[:split]),pin_memory=True, num_workers=1)

        valid_queue = DataLoader(
        train_data, batch_size=args.batch_size, sampler=SubsetRandomSampler(
              indices[split:num_train]),pin_memory=True, num_workers=5)
      else:
        train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=1)

        valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=1)
    elif args.dataset == 'vegetables':
      cs_trainandval_path = "/home/lwj-hdu/shz/2109/20211110/trainandval"
      train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(cs_trainandval_path)
      data_transform = {
          "train": transforms.Compose([
                                      transforms.Resize(380),
                                      # transforms.Resize(600),
                                      transforms.CenterCrop(256),
                                      # transforms.CenterCrop(450),
                                      transforms.RandomHorizontalFlip(0.5),
                                      transforms.RandomRotation(0.3),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
          "val": transforms.Compose([
                                    # transforms.Resize(600),
                                    transforms.Resize(380),
                                    transforms.CenterCrop(256),
                                    # transforms.CenterCrop(450),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
      train_data = MyDataSet(images_path=train_images_path,images_class=train_images_label,transform=data_transform["train"])
      valid_data = MyDataSet(images_path=val_images_path,images_class=val_images_label,transform=data_transform["val"])
      if args.if_search == True:
        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(args.train_portion * num_train))
        train_queue = DataLoader(
        train_data, batch_size=args.batch_size, sampler=SubsetRandomSampler(indices[:split]), pin_memory=True, num_workers=1)

        valid_queue = DataLoader(
        train_data, batch_size=args.batch_size, sampler=SubsetRandomSampler(
              indices[split:num_train]), pin_memory=True, num_workers=1)
      else:
        train_queue = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=5,collate_fn=train_data.collate_fn)

        valid_queue = DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=5,collate_fn=valid_data.collate_fn)
    return train_queue, valid_queue

def read_split_data(root: str, val_rate: float = 0.3):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)


    food_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]

    food_class.sort()

    class_indices = dict((k, v) for v, k in enumerate(food_class))


    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型

    for cla in food_class:
        cla_path = os.path.join(root, cla)

        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]

        image_class = class_indices[cla]

        every_class_num.append(len(images))

        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    
    return train_images_path, train_images_label, val_images_path, val_images_label
class MyDataSet(Dataset):

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


def calc_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    t, h = divmod(h, 24)
    return {'day': t, 'hour': h, 'minute': m, 'second': int(s)}

# gene_reduce = parse(F.sigmoid(self.alphas_reduce).data.cpu().numpy(), PRIMITIVES, self.op_threshold, self.parse_method, self._steps)
# gene_normal = parse(F.sigmoid(self.alphas_normal).data.cpu().numpy(), PRIMITIVES, self.op_threshold, self.parse_method, self._steps)
def parse(weights, operation_set,
           op_threshold, parse_method, steps):
  gene = []
  if parse_method == 'darts':
    n = 2
    start = 0
    for i in range(steps): # step = 4
      end = start + n
      W = weights[start:end].copy()
      edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]
      for j in edges:
        k_best = None
        for k in range(len(W[j])):
            if k_best is None or W[j][k] > W[j][k_best]:
              k_best = k
        gene.append((operation_set[k_best], j)) # geno item : (operation, node idx)
      start = end
      n += 1
  elif 'threshold' in parse_method:
    n = 2
    start = 0
    for i in range(steps): # step = 4
      end = start + n
      W = weights[start:end].copy()
      if 'edge' in parse_method:
        edges = list(range(i + 2))
      else: # select edges using darts methods
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]

      for j in edges:
        if 'edge' in parse_method: # OP_{prob > T} AND |Edge| <= 2
          topM = sorted(enumerate(W[j]), key=lambda x: x[1])[-2:]
          for k, v in topM: # Get top M = 2 operations for one edge
            if W[j][k] >= op_threshold:
              gene.append((operation_set[k], i+2, j))
        elif 'sparse' in parse_method: # max( OP_{prob > T} ) and |Edge| <= 2
          k_best = None
          for k in range(len(W[j])):
            if k_best is None or W[j][k] > W[j][k_best]:
              k_best = k
          if W[j][k_best] >= op_threshold:
            gene.append((operation_set[k_best], i+2, j))
        else:
            raise NotImplementedError("Not support parse method: {}".format(parse_method))
      print(W)
      start = end
      n += 1
  return gene
