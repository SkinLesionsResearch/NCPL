from __future__ import absolute_import
from .inception import *
from .resnet import *
from .vgg import *
from .googlenet import *
from .alexnet import *
from .senet import *

__factory = {
    'ResBase': ResBase,
    'InceptionBase': InceptionBase,
    'VGGBase': VGGBase,
    'GoogLeNet': GoogLeNet,
    'AlexNet': AlexNet,
	'SEInception3': SEInception3
}
