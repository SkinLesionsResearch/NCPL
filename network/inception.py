import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3

class InceptionBase(nn.Module):

    def __init__(self, num_classes):
        super(InceptionBase, self).__init__()
        model_inception = inception_v3(pretrained=True, aux_logits=False)
        self.Conv2d_1a_3x3 = model_inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model_inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model_inception.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model_inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model_inception.Conv2d_4a_3x3
        self.Mixed_5b = model_inception.Mixed_5b
        self.Mixed_5c = model_inception.Mixed_5c
        self.Mixed_5d = model_inception.Mixed_5d
        self.Mixed_6a = model_inception.Mixed_6a
        self.Mixed_6b = model_inception.Mixed_6b
        self.Mixed_6c = model_inception.Mixed_6c
        self.Mixed_6d = model_inception.Mixed_6d
        self.Mixed_6e = model_inception.Mixed_6e
        self.Mixed_7a = model_inception.Mixed_7a
        self.Mixed_7b = model_inception.Mixed_7b
        self.Mixed_7c = model_inception.Mixed_7c
        self.fc = nn.Linear(model_inception.fc.in_features, 1024)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        x = F.dropout(x, training=self.training)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        features = self.fc(x)
        logits = self.classifier(features)
        return features, logits

if __name__ == '__main__':
    device = torch.device('cuda')
    # model = InceptionBase().to(device)
    model = InceptionBase(num_classes=3).to(device)
    print(model)
