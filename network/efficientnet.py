from efficientnet_pytorch import EfficientNet
VALID_MODELS = (
    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
    'efficientnet-b8',
    # Support the construction of 'efficientnet-l2' without pretrained weights
    'efficientnet-l2'
)
effi_dict = {'efficientnet-b0': EfficientNet.from_pretrained('efficientnet-b0'),
            'efficientnet-b1': EfficientNet.from_pretrained('efficientnet-b1'),
            'efficientnet-b2': EfficientNet.from_pretrained('efficientnet-b2'),
            'efficientnet-b3': EfficientNet.from_pretrained('efficientnet-b3'),
            'efficientnet-b4': EfficientNet.from_pretrained('efficientnet-b4'),
            'efficientnet-b5': EfficientNet.from_pretrained('efficientnet-b5'),
            'efficientnet-b6': EfficientNet.from_pretrained('efficientnet-b6'),
            'efficientnet-b7': EfficientNet.from_pretrained('efficientnet-b7'),
            'efficientnet-l2': EfficientNet.from_pretrained('efficientnet-b8'),
            }

class EfficientBase(nn.Module):
    def __init__(self, name, num_classes):
        super(EfficientBase, self).__init__()
        model_efficient = effi_dict[name](pretrained=True)


if __name__ == '__main__':
    model = EfficientNet.from_pretrained('efficientnet-b0')
    print(model)
