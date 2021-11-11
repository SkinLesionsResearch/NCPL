import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

vgg_dict = {"vgg11":models.vgg11, "vgg13":models.vgg13, "vgg16":models.vgg16, "vgg19":models.vgg19,
"vgg11bn":models.vgg11_bn, "vgg13bn":models.vgg13_bn, "vgg16bn":models.vgg16_bn, "vgg19bn":models.vgg19_bn}

class VGGBase(nn.Module):
  def __init__(self, name, num_classes):
    super(VGGBase, self).__init__()
    self.model_vgg = vgg_dict[name](pretrained=True)
    in_features = self.model_vgg.classifier[6].in_features
    self.model_vgg.classifier = self.model_vgg.classifier[:-1]
    # self.classifier = nn.Linear(in_features, num_classes)

    # self.classifier = nn.Sequential(
    #     nn.Linear(256 * 6 * 6, 4096),
    #     nn.ReLU(inplace=True),
    #     nn.Linear(4096, 4096),
    #     nn.ReLU(inplace=True),
    #     nn.Linear(4096, num_classes),
    # )
    self.classifier = nn.Sequential(
        nn.ReLU(inplace=True),
        nn.Linear(in_features, num_classes),
    )

  def forward(self, x):
      features = self.model_vgg(x)
      logits = self.classifier(features)
      return features, logits

if __name__ == '__main__':
    model = VGGBase('vgg11', 2).cuda()
    import torch
    rand = torch.rand(3, 3, 256, 256).cuda()
    out = model(rand)
    print(out[0].shape, ", ", out[1].shape)