import torch.nn as nn
from torchvision import models

class AlexNet(nn.Module):
  def __init__(self, num_classes):
    super(AlexNet, self).__init__()
    self.model_alexnet = models.alexnet(pretrained=True)
    in_features = self.model_alexnet.classifier[6].in_features
    self.model_alexnet.classifier = self.model_alexnet.classifier[:-1]
    self.fc = nn.Linear(in_features, num_classes)

  def forward(self, x):
      features = self.model_alexnet(x)
      logits = self.fc(features)
      return features, logits

if __name__ == '__main__':
    model = AlexNet(2).cuda()
    print(model)
    import torch
    rand = torch.rand(3, 3, 256, 256).cuda()
    out = model(rand)
    print(out[0].shape)