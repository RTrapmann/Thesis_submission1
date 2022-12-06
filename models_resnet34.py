import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as  nn
import torch.nn.functional as F
from functools import partial
from dataclasses import dataclass
from collections import OrderedDict

class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size
        
conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)     
conv = conv3x3(in_channels=32, out_channels=64)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels =  in_channels, out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()   
    
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(OrderedDict(
        {
            'conv' : nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            'bn' : nn.BatchNorm2d(self.expanded_channels)
            
        })) if self.should_apply_shortcut else None
        
        
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels        

from collections import OrderedDict
def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs), 
                          'bn': nn.BatchNorm2d(out_channels) }))

class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation(),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )
class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
           conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
             activation(),
             conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
             activation(),
             conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
        )
    
class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion, 
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x

class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    """
    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], deepths=[2,2,2,2], 
                 activation=nn.ReLU, block=ResNetBasicBlock, *args,**kwargs):
        super().__init__()
        
        self.blocks_sizes = blocks_sizes
        
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([ 
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation, 
                        block=block,  *args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion, 
                          out_channels, n=n, activation=activation, 
                          block=block, *args, **kwargs) 
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]       
        ])
        
        
    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x

class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x

class ResNet(nn.Module):
    name = "ResNet 34"
    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def ResNet34(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBasicBlock, deepths=[3, 4, 6, 3])


# class Bottleneck(nn.Module):
#     expansion = 4
#     def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
#         super(Bottleneck, self).__init__()
        
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
#         self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
#         self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
#         self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
#         self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
#         self.i_downsample = i_downsample
#         self.stride = stride
#         self.relu = nn.ReLU()
        
#     def forward(self, x):
#         identity = x.clone()
#         x = self.relu(self.batch_norm1(self.conv1(x)))
        
#         x = self.relu(self.batch_norm2(self.conv2(x)))
        
#         x = self.conv3(x)
#         x = self.batch_norm3(x)
        
#         #downsample if needed
#         if self.i_downsample is not None:
#             identity = self.i_downsample(identity)
#         #add identity
#         x+=identity
#         x=self.relu(x)
        
#         return x

# class Block(nn.Module):
#     expansion = 1
#     def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
#         super(Block, self).__init__()
       

#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
#         self.batch_norm1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
#         self.batch_norm2 = nn.BatchNorm2d(out_channels)

#         self.i_downsample = i_downsample
#         self.stride = stride
#         self.relu = nn.ReLU()

#     def forward(self, x):
#       identity = x.clone()

#       x = self.relu(self.batch_norm2(self.conv1(x)))
#       x = self.batch_norm2(self.conv2(x))

#       if self.i_downsample is not None:
#           identity = self.i_downsample(identity)
#       print(x.shape)
#       print(identity.shape)
#       x += identity
#       x = self.relu(x)
#       return x

# class ResNet(nn.Module):
#     name = "ResNet50"

#     def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
#         super(ResNet, self).__init__()
#         self.in_channels = 64
        
#         self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.batch_norm1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU()
#         self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
#         self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
#         self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
#         self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
#         self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
#         self.avgpool = nn.AdaptiveAvgPool2d((1,1))
#         self.fc = nn.Linear(512*ResBlock.expansion, num_classes)
        
#     def forward(self, x):
#         x = self.relu(self.batch_norm1(self.conv1(x)))
#         x = self.max_pool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
        
#         x = self.avgpool(x)
#         x = x.reshape(x.shape[0], -1)
#         x = self.fc(x)
        
#         return x
        
#     def _make_layer(self, ResBlock, blocks, planes, stride=1):
#         ii_downsample = None
#         layers = []
        
#         if stride != 1 or self.in_channels != planes*ResBlock.expansion:
#             ii_downsample = nn.Sequential(
#                 nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
#                 nn.BatchNorm2d(planes*ResBlock.expansion)
#             )
            
#         layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
#         self.in_channels = planes*ResBlock.expansion
        
#         for i in range(blocks-1):
#             layers.append(ResBlock(self.in_channels, planes))
            
#         return nn.Sequential(*layers)

        
        
# # def ResNet50(num_classes = 43, channels=3):
# #     return ResNet(Bottleneck, [3,4,6,3], num_classes, channels)

# def ResNet101(num_classes = 43, channels=3):
#     return ResNet(Bottleneck, [3,4,23,3], num_classes, channels)





# Conv NN constructors:
class ConvNN(nn.Module):
    name = ""

    def __init__(self, params):
        """
        The purpose in using this class in the way it built is to make the process of creating CNNs with the ability
        to control its capacity in efficient way (programming efficiency - not time efficiency).
        I found it very useful in constructing experiments. I tried to make this class general as possible.
        :param params: a dictionary with the following attributes:
            capacity influence:
            - channels_lst: lst of the channels sizes. the network complexity is inflected mostly by this parameter
              * for efficiency channels_lst[0] is the number of input channels
            - #FC_Layers: number of fully connected layers
            - extras_blocks_components: in case we want to add layers from the list ["dropout", "max_pool", "batch norm"]
                                        to each block we can do it. Their parameters are attributes of this dict also.
              * notice that if max_pool in extras_blocks_components then we reduce dims using max_pool instead conv
                layer (the conv layer will be with stride 1 and padding)
            - p_dropout: the dropout parameter

            net structure:
            - in_wh: input width and height
        """
        super().__init__()
        self.params = params
        channels_lst = params["channels_lst"]
        extras_block_components = params["extras_blocks_components"]

        assert 2 <= len(channels_lst) <= 5
        conv_layers = []
        for i in range(1, len(channels_lst)):
            """
            Dims calculations: next #channels x (nh-filter_size/2)+1 x (nw-filter_size/2)+1
            """
            filter_size, stride, padding = (4, 2, 1) if "max_pool" not in extras_block_components else (5, 1, 2)
            conv_layers.append(nn.Conv2d(channels_lst[i - 1], channels_lst[i], filter_size, stride, padding, bias=False))
            conv_layers.append(params["activation"]())

            for comp in extras_block_components:
                if comp == "dropout":
                    conv_layers.append(nn.Dropout(params["p_dropout"]))
                if comp == "max_pool":
                    conv_layers.append(nn.MaxPool2d(2, 2))
                if comp == "batch_norm":
                    conv_layers.append(nn.BatchNorm2d(channels_lst[i]))

        out_channels = channels_lst[-1]
        if params["CNN_out_channels"] is not None:
            conv_layers.append(nn.Conv2d(channels_lst[-1], params["CNN_out_channels"], 1))
            conv_layers.append(params["activation"]())
            out_channels = params["CNN_out_channels"]

        self.cnn = nn.Sequential(*conv_layers)

        lin_layers = []
        wh = params["in_wh"] // (2 ** (len(channels_lst) - 1))  # width and height of last layer output
        lin_layer_width = out_channels * (wh ** 2)
        for _ in range(params["#FC_Layers"] - 1):
            lin_layers.append(nn.Linear(lin_layer_width, lin_layer_width))
        lin_layers.append(nn.Linear(lin_layer_width, params["out_size"]))
        self.linear_nn = nn.Sequential(*lin_layers)

        """ we use CE loss so we don't need to apply softmax (for test loss we also use the same CE. for accuracy
            we choose the highest value - this property is saved under softmax)"""

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.view(1, *x.shape)
        assert x.shape[2] == x.shape[3] == self.params["in_wh"]
        assert x.shape[1] == self.params["channels_lst"][0]

        cnn_output = self.cnn(x).view((x.shape[0], -1))
        lin_output = self.linear_nn(cnn_output)
        return lin_output


def create_conv_nn(params):
    return ConvNN(params)


# class CNNTrafficSignNet(nn.Module):
#     name = "CNN-TrafficSignNet"

#     def __init__(self, in_channels=3, out_channels=43):
#         super().__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv2d(in_channels, 50, 3, padding=1, stride=2),  # 32
#             nn.ReLU(),
#             nn.Dropout(0.05),
#             # nn.MaxPool2d(2, stride=2),  # 16
#             nn.Conv2d(50, 100, 3, padding=1, stride=2),  # 16
#             nn.ReLU(),
#             nn.Dropout(0.05),
#             # nn.MaxPool2d(2, stride=2),  # 8
#             nn.Conv2d(100, 150, 5),  # 4
#             nn.ReLU(),
#             nn.Conv2d(150, 75, 3, padding=1),  # 4
#             nn.ReLU(),
#             nn.Conv2d(75, 50, 1),
#             nn.ReLU(),
#         )

#         self.lin = nn.Sequential(
#             nn.Linear(50 * 4 * 4, 100),
#             nn.ReLU(),
#             # nn.Dropout(0.1),
#             nn.Linear(100, out_channels)
#         )

#     def forward(self, x):
#         x = self.cnn(x)
#         x = x.view(-1, 50 * 4 * 4)
#         x = self.lin(x)
#         return x

#
#
# class STN1D(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Spatial transformer localization-network
#         self.localization = nn.Sequential(
#             nn.Conv2d(1, 8, kernel_size=7),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True),
#             nn.Conv2d(8, 10, kernel_size=5),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True)
#         )
#
#         # Regressor for the 3 * 2 affine matrix
#         self.fc_loc = nn.Sequential(
#             nn.Linear(10 * 3 * 3, 32),
#             nn.ReLU(True),
#             nn.Linear(32, 3 * 2)
#         )
#
#         # Initialize the weights/bias with identity transformation
#         self.fc_loc[2].weight.data.zero_()
#         self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
#
#     # Spatial transformer network forward function
#     def forward(self, x):
#         xs = self.localization(x)
#         xs = xs.view(-1, 10 * 3 * 3)
#         theta = self.fc_loc(xs)
#         theta = theta.view(-1, 2, 3)
#
#         grid = F.affine_grid(theta, x.size())
#         x = F.grid_sample(x, grid)
#
#         return x
#
# class STN_MNISTNet(nn.Module):
#     name = "MNIST-NET"
#
#     def __init__(self):
#         super().__init__()
#         self.stn = STN1D()
#         self.cnn = nn.Sequential(
#             nn.Conv2d(1, 100, 3, padding=1),  # 28
#             nn.ReLU(),
#             nn.Conv2d(100, 100, 3, padding=1),  # 28
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.MaxPool2d(2, stride=2),  # 14
#             nn.Conv2d(100, 200, 3, padding=1),  # 14
#             nn.ReLU(),
#             nn.Conv2d(200, 200, 3, padding=1),  # 14
#             nn.Dropout(0.1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, stride=2),  # 7
#             nn.Conv2d(200, 400, 5),  # 3
#             nn.ReLU(),
#             nn.Conv2d(400, 250, 3, padding=1),  # 3
#             nn.ReLU(),
#             nn.Conv2d(250, 100, 1),
#             nn.ReLU(),
#         )
#
#         self.lin = nn.Sequential(
#             nn.Linear(100 * 3 * 3, 350),
#             nn.Dropout(0.1),
#             nn.ReLU(),
#             nn.Linear(350, 10)
#         )
#
#     def forward(self, x):
#         x = self.stn(x)
#         x = self.cnn(x)
#         x = x.view(-1, 100 * 3 * 3)
#         x = self.lin(x)
#         return x
#
class Stn(nn.Module):
    """ from https://github.com/wolfapple/traffic-sign-recognition/blob/master/notebook.ipynb"""

    def __init__(self):
        super().__init__()
        # localization network
        self.loc_net = nn.Sequential(
            nn.Conv2d(3, 50, 7),
            nn.MaxPool2d(2, 2),
            nn.ELU(),
            nn.Conv2d(50, 100, 5),
            nn.MaxPool2d(2, 2),
            nn.ELU()
        )
        # regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(100 * 4 * 4, 100),
            nn.ELU(),
            nn.Linear(100, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.loc_net(x)
        xs = xs.view(-1, 100 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x



class STNTrafficSignNet(nn.Module):
    name = "STN-TrafficSignNet"

    def __init__(self, in_channels=3, out_channels=43):
        super().__init__()
        self.stn = Stn()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 100, 3, padding=1),  # 32
            nn.ReLU(),
            nn.Conv2d(100, 100, 3, padding=1),  # 32
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, stride=2),  # 16
            nn.Conv2d(100, 200, 3, padding=1),  # 16
            nn.ReLU(),
            nn.Conv2d(200, 200, 3, padding=1),  # 16
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2, stride=2),  # 8
            nn.Conv2d(200, 400, 5),  # 4
            nn.ReLU(),
            nn.Conv2d(400, 250, 3, padding=1),  # 4
            nn.ReLU(),
            nn.Conv2d(250, 100, 1),
            nn.ReLU(),
        )

        self.lin = nn.Sequential(
            nn.Linear(100 * 4 * 4, 350),
            nn.ReLU(),
            nn.Linear(350, out_channels)
        )

    def forward(self, x):
        x = self.stn(x)
        x = self.cnn(x)
        x = x.view(-1, 100 * 4 * 4)
        x = self.lin(x)
        return x