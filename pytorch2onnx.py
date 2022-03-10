import cv2
import time
import torch
import numpy as np
from torch.nn import DataParallel

from collections import OrderedDict
import torchsummary
from torchvision import transforms as T
import onnxruntime as rt
 
import argparse
import os
import sys
import time
import torch.utils
import torchvision.datasets as dset
from thop import profile
import teachermodel.vgg,teachermodel.densenet,teachermodel.efficentnet,teachermodel.inceptionnet,teachermodel.resnet
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
parser.add_argument('--student_arch', type=str, default='resnet50_stu', help='choose model type')
parser.add_argument('--model', type=str, default='resnet50', help='choose model type')



args = parser.parse_args()
args.auto_aug = False
args.cutout = False
args.auxiliary = True





        
    # torch2onnx(model, "../onnxmodel/densenet201_vege.onnx")
def torch2onnx(model, save_path):
    model.eval()
    data = torch.rand(1,3,256,256)
    input_names = ['input']
    output_names = ['out']
    torch.onnx.export(model,
                      data,
                      save_path,
                      export_params=True,
                      opset_version=11,
                      input_names=input_names,
                      output_names=output_names)
    print("torch2onnx finish")
 
 
def img_process(img_path):
    # normalize = T.Normalize(mean = [0.5, 0.5, 0.5],
    #                         std = [0.5, 0.5, 0.5])
    normalize = T.Normalize(mean = [0.485, 0.456, 0.406],
                            std = [0.229, 0.224, 0.225])
    transforms = T.Compose([T.ToTensor(),
                            normalize])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = transforms(img)
    img = img.unsqueeze(0)
    return img
 
 
def onnx_runtime(img):
    # sess = rt.InferenceSession("../onnxmodel/{}-{}.onnx".format(args.student_arch,args.dataset))
    sess = rt.InferenceSession("../onnxmodel/densenet201_vege.onnx")
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    
    t0 = time.time()
    for i in range(100):
        pred_onnx = sess.run([output_name], {input_name:np.array(img)})
    t1 = time.time()
    print("用onnx完成100次推理消耗的时间:%s" % (t1-t0))
    print("用onnx推理的结果如下：")
    cls = np.argmax(pred_onnx[0][0], axis=0) #输出的是类
    print(cls)
 
 
def model_load(model_pth):
    state_dict = torch.load(model_pth, map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith("module."):
            name = name[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    return model
 
 
if __name__ == "__main__":

    model = dutils.get_network(args)
    weights = dutils.get_teacher_weight(args)
    model.load_state_dict(torch.load(
    weights,map_location='cpu'),strict=False)  # 仅加载参数
    # model = dutils.get_student(args)
    # model.eval()
    # summary(model, input_size=(3, 32, 32), batch_size=1,device="cpu")
    img = img_process("./0626382.jpg")
    # print()
    t0 = time.time()
    for i in range(100):
        outputs = model(img)
    t1 = time.time()
    print("用pytorch完成100次推理消耗的时间:%s" % (t1-t0))
    print("用pytorch推理的结果如下：")
    print(np.argmax(outputs[0].detach().tolist()))
    print()

    # torch2onnx(model, "../onnxmodel/{}-{}.onnx".format(args.student_arch,args.dataset))
    torch2onnx(model, "../onnxmodel/densenet201_vege.onnx")
    onnx_runtime(img)