import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_utils


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
        print("pos_nan")
    if torch.isnan(loss_neg):
        print("neg_nan")

    return loss_pos + loss_neg
    # return F.cross_entropy(logits, labels)
