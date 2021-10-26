import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_utils
from torch.autograd import Variable


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def uncertainty_loss(args, logits, labels, pos_threshold=0.8, neg_threshold=0.3):
    soft_mask_output = nn.Softmax(dim=1)(logits)

    pos_idx = torch_utils.get_idx(soft_mask_output, soft_mask_output >= pos_threshold)[0]
    neg_idx = torch_utils.get_idx(soft_mask_output, soft_mask_output <= neg_threshold)[0]

    loss_pos = torch.tensor(0.0).to(device=args.device)
    loss_neg = torch.tensor(0.0).to(device=args.device)

    if sum(pos_idx * 1) > 0:
        loss_pos = F.cross_entropy(logits[pos_idx], labels[pos_idx], reduction="mean")
    if sum(neg_idx * 1) > 0:
        neg_soft_mask_output = soft_mask_output[neg_idx]
        neg_outputs = torch.clamp(neg_soft_mask_output, 1e-7, 1.0)
        neg_logits = logits[neg_idx]
        neg_mask_labels = torch.where(neg_soft_mask_output <= neg_threshold,
                                      torch.ones_like(neg_soft_mask_output),
                                      torch.zeros_like(neg_soft_mask_output))
        y_neg = torch.ones(neg_logits.shape).to(device=args.device, dtype=logits.dtype)
        loss_neg = torch.mean((-torch.sum(y_neg * torch.log(1 - neg_outputs) * neg_mask_labels, dim=-1)) /
                              (torch.sum(neg_mask_labels, dim=-1) + 1e-7))
    if torch.isnan(loss_pos):
        loss_pos = torch.tensor(0.0).to(device=args.device)
        print("pos_nan")
    if torch.isnan(loss_neg):
        loss_neg = torch.tensor(0.0).to(device=args.device)
        print("neg_nan")

    return loss_pos + loss_neg
    # return F.cross_entropy(logits, labels)

class FocalLossClassPropotion(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \class_propotion (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            class_propotion(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """

    def __init__(self, class_num, class_propotion=None,
                 alpha=1.1, class_factor=torch.Tensor([1, 1, 1, 1, 1, 1, 1]),
                 gamma=2, size_average=True):
        super(FocalLossClassPropotion, self).__init__()
        if class_propotion is None:
            self.class_propotion = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(class_propotion, Variable):
                self.class_propotion = class_propotion
            else:
                self.class_propotion = Variable(class_propotion) if isinstance(class_propotion, torch.Tensor) \
                                                                 else Variable(torch.tensor(class_propotion))
        self.alpha = alpha
        self.class_factor = class_factor
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.class_propotion.is_cuda:
            self.class_propotion = self.class_propotion.cuda()
        if inputs.is_cuda and not self.class_factor.is_cuda:
            self.class_factor = self.class_factor.cuda()
        batch_size = inputs.size(0)
        class_factor = self.class_factor.repeat([batch_size, 1])
        class_propotion = self.class_propotion.repeat([batch_size, 1])
        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -torch.pow(class_propotion, self.alpha) * class_factor \
                     * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


