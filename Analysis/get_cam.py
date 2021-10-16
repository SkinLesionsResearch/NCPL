# coding=utf-8
# author: huang.rong
# date: 2021/04/23

from __future__ import print_function, division
import os
import time
import copy
import sys
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchnet import meter
import torchvision
from torchvision import models, transforms, datasets
import cv2
import pandas as pd
import numpy as np


# ******************省略加载数据集的工作，自行补全******************************************


########若使用现成的模型则直接看XXNet 在forward部分添加finalconv，即需要可视化的层#####################################
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

class ResNet101(nn.Module):

    def __init__(self, layers):
        super(ResNet101, self).__init__()
        self.layers = layers
        self.conv = nn.Sequential(nn.Conv2d(2048, 512, 3, padding=1, bias=False),
                    nn.BatchNorm2d(512), nn.ReLU())
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, 512, 1))
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        x = self.layers(x)
        x = self.conv(x)
        self.finalconv = x.detach()       #  可视化的关键，一般是卷积层的最后一层！！！！！！！！！！！！！！！！
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# show tensor images
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    # plt.show()      # decide on whether your compiler is interactive
    plt.pause(0.001)  # pause a bit so that plots are updated



model_layer = ResNet(Bottleneck, [3, 4, 23, 3])
model_dict = model_layer.state_dict()
pretrained_dict = models.resnet101(pretrained=True).state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model_layer.load_state_dict(model_dict)  #backbone即ResNet101加载在ImageNet上预训练好的参数（迁移学习）
model = ResNet101(model_layer)
model = model.to(device)
#model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))  #加载已经训练好的需要可视化的模型

# get CAM
# *****************************CAM*****************************************************
def returnCAM(feature_conv, weight_softmax, class_idx):
    bz, nc, h, w = feature_conv.shape  # 1,2048,7,7
    output_cam = []
    for idx in class_idx:  # 若只输出预测概率最大值结果不需要for循环
        feature_conv = feature_conv.reshape((nc, h * w))
        cam = weight_softmax[idx].dot(
            feature_conv.reshape((nc, h * w)))  # (2048, ) * (2048, 7*7) -> (7*7, ) （n,）是一个数组，既不是行向量也不是列向量
        cam = cam.reshape(h, w)
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize
        cam_img = np.uint8(255 * cam_img)  # Format as CV_8UC1 (as applyColorMap required)
        output_cam.append(cam_img)
    return output_cam


if not os.path.exists(cam_save_path):
    os.mkdir(cam_save_path)

all_data = pd.read_excel(info_xlsx[predict_mode], engine='openpyxl')
image_list = all_data['filename'].values.tolist()
label_list = all_data['label'].values.tolist()
all_data_dict = dict(zip(image_list, label_list))
images_path_list = [os.path.join(data_path, img) for img in image_list]

model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

# get weight matrix of full connection
fc_weights = model.state_dict()['fc.weight'].cpu().numpy()  # [2,2048]

# set the model to eval mode, it's necessary
model.eval()

for i, img_path in enumerate(images_path_list):
    print('*' * 10)
    # img_path = './bee.jpg'  # test single image
    _, img_name = os.path.split(img_path)
    img_label = class_names[all_data_dict[img_name]]
    img = Image.open(img_path).convert('RGB')
    img_tensor = data_transforms['test'](img).unsqueeze(0)  # [1,3,224,224]
    inputs = img_tensor.to(device)

    logit = model(inputs)  # [1,2] -> [ 3.3207, -2.9495]
    h_x = torch.nn.functional.softmax(logit, dim=1).data.squeeze()  # tensor([0.9981, 0.0019])
    probs, idx = h_x.sort(0, True)  # sorted in descending order

    probs = probs.cpu().numpy()  # if tensor([0.0019,0.9981]) ->[0.9981, 0.0019]
    idx = idx.cpu().numpy()  # [1, 0]
    for id in range(2):
        # 0.559 -> neg, 0.441 -> pos
        print('{:.3f} -> {}'.format(probs[id], class_names[idx[id]]))

    features = model.finalconv.cpu().numpy()  # [1,2048,7,7]
    print('final feature map layer shape is ', features.shape)
    CAMs = returnCAM(features, fc_weights, [idx[0]])  # output the most probability class activate map
    print(img_name + ' output for the top1 prediction: %s' % class_names[idx[0]])
    img = cv2.imread(img_path)
    height, width, _ = img.shape  # get input image size
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)),
                                cv2.COLORMAP_JET)  # CAM resize match input image size
    heatmap[np.where(CAMs[0] <= 100)] = 0
    result = heatmap * 0.3 + img * 0.5  # ratio

    text = '%s %.2f%%' % (class_names[idx[0]], probs[0] * 100)
    cv2.putText(result, text, (210, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9,
                color=(123, 222, 238), thickness=2, lineType=cv2.LINE_AA)

    image_name_ = img_name.split(".")[-2]
    cv2.imwrite(cam_save_path + r'/' + image_name_ + '_' + 'pred_' + class_names[idx[0]] + '.jpg', result)
