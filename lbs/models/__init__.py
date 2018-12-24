"""
The models subpackage contains all the various models that we test on,
as well as a generic method that lets us create the models by name.
"""
from absl import flags
from .wide_resnet import WideResNet
from .lenet import LeNet
from .rnn import RNNModel
from .drn import drn_d_22
from .resnet import resnet18, resnet34, resnet50
from .densenet import DenseNet121
from .mobilenetv2 import MobileNetV2
from .vgg import VGG as vgg
from .alexnet import AlexNet

flags.DEFINE_float(
    'dropout', 0.0, 'Dropout rate for models that already have dropout layers'
    'e.g. the lstm.')

_LSTM_MODELS = ['lstm']


def needs_hidden_state(model_name):
    """
    Returns true if the model ID'd by model_name
    is recurrent and requires a hidden state
    """
    return model_name in _LSTM_MODELS


def build_model(name, num_classes=10, vocab_len=1000):
    """
    Generates model corresponding to the given name
    returns: (network, boolean if network stores hidden state (LSTM))
    """
    if name == 'wideresnet28':
        model = WideResNet(
            depth=28, widen_factor=10, dropout=0.3, num_classes=num_classes)
    elif name == 'lenet':
        model = LeNet(num_classes)
    elif name == 'vgg':
        model = vgg(num_classes=num_classes, vgg_name='VGG16')
    elif name == 'lstm':
        model = RNNModel(
            rnn_type='LSTM',
            ntoken=vocab_len,
            ninp=650,
            nhid=650,
            nlayers=2,
            dropout=flags.FLAGS.dropout,
            tie_weights=True)
    elif name == 'alexnet':
        model = AlexNet(num_classes=num_classes)
    elif name == 'densenet121':
        model = DenseNet121(num_classes=num_classes)
    elif name == 'mobilenetv2':
        model = MobileNetV2(num_classes=num_classes)
    elif name == 'resnet18':
        model = resnet18(num_classes=num_classes)
    elif name == 'resnet34':
        model = resnet34(num_classes=num_classes)
    elif name == 'resnet50':
        model = resnet50(num_classes=num_classes)
    elif name == 'drn_d_22':
        model = drn_d_22(num_classes=num_classes)
    else:
        raise ValueError('unsupported model {}'.format(name))
    return model, needs_hidden_state(name)
