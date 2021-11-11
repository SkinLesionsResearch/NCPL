import torch
import torch.nn as nn
from torchvision.models import googlenet

class GoogLeNet(nn.Module):

    def __init__(self, num_classes):
        super(GoogLeNet, self).__init__()
        model_googlenet = googlenet(pretrained=True, aux_logits=False)
        self.conv1 = model_googlenet.conv1
        self.maxpool1 = model_googlenet.maxpool1
        self.conv2 = model_googlenet.conv2
        self.conv3 = model_googlenet.conv3
        self.maxpool2 = model_googlenet.maxpool2

        self.inception3a = model_googlenet.inception3a
        self.inception3b = model_googlenet.inception3b
        self.maxpool3 = model_googlenet.maxpool3

        self.inception4a = model_googlenet.inception4a
        self.inception4b = model_googlenet.inception4b
        self.inception4c = model_googlenet.inception4c
        self.inception4d = model_googlenet.inception4d
        self.inception4e = model_googlenet.inception4e
        self.maxpool4 = model_googlenet.maxpool4

        self.inception5a = model_googlenet.inception5a
        self.inception5b = model_googlenet.inception5b

        self.avgpool = model_googlenet.avgpool
        self.fc = nn.Linear(model_googlenet.fc.in_features, num_classes)

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)
        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7
        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        features = torch.flatten(x, 1)
        # N x 1024
        logits = self.fc(features)
        return features, logits


if __name__ == '__main__':
    model_googlenet = GoogLeNet(num_classes=3).cuda()
    print(model_googlenet)
    # rand = torch.rand(32, 3, 256, 256).cuda()
    # f, l = model_googlenet(rand)
    # print(f.shape, l.shape)