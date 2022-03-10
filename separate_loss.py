import torch
import torch.nn as nn
import torch.nn.functional as F

#可选正则项 [l1,l2] aux_loss_weight 论文默认是10.0
__all__ = ['ConvSeparateLoss', 'TriSeparateLoss']
# l2
class ConvSeparateLoss(nn.modules.loss._Loss):
    """Separate the weight value between each operations using L2"""
    def __init__(self, weight=0.1, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(ConvSeparateLoss, self).__init__(size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, input1, target1, input2):
        loss1 = F.cross_entropy(input1, target1)
        loss2 = -F.mse_loss(input2, torch.tensor(0.5, requires_grad=False).cuda())
        return loss1 + self.weight*loss2, loss1.item(), loss2.item()

# l1
class TriSeparateLoss(nn.modules.loss._Loss):
    """Separate the weight value between each operations using L1"""
    def __init__(self, weight=0.1, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(TriSeparateLoss, self).__init__(size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, input1, target1, input2):
        loss1 = F.cross_entropy(input1, target1)
        loss2 = -F.l1_loss(input2, torch.tensor(0.5, requires_grad=False).cuda())
        return loss1 + self.weight*loss2, loss1.item(), loss2.item()

# KDloss
class KD_loss(nn.modules.loss._Loss):
  def __init__(self,  weight=0.1,alpha=0.8,T = 10,size_average=None, ignore_index=-100,reduce=None, reduction: str = 'mean') -> None:
      super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)
      self.ignore_index = ignore_index
      self.weight = weight
      self.alpha = alpha
      self.T = T

  def forward(self, studentout, target, input2,teacherout):
    loss1 = F.cross_entropy(studentout, target) #input1是studentmodel输出

    loss2 = -F.mse_loss(input2, torch.tensor(0.5, requires_grad=False).cuda())
    loss3 = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(studentout / self.T, dim=1),
                          F.softmax(teacherout / self.T, dim=1)) 
    return loss3 *(self.alpha * self.T * self.T) + loss1 * (1. - self.alpha) + self.weight*loss2, loss1.item(), loss2.item(),loss3.item()
        


