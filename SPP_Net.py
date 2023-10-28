from math import floor, ceil
import torch
import torch.nn as nn
import torch.nn.functional as F
 
class SpatialPyramidPooling2d(nn.Module):
    r"""apply spatial pyramid pooling over a 4d input(a mini-batch of 2d inputs 
    with additional channel dimension) as described in the paper
    'Spatial Pyramid Pooling in deep convolutional Networks for visual recognition'
    Args:
        num_level:
        pool_type: max_pool, avg_pool, Default:max_pool
    By the way, the target output size is num_grid:
        num_grid = 0
        for i in range num_level:
            num_grid += (i + 1) * (i + 1)
        num_grid = num_grid * channels # channels is the channel dimension of input data
    examples:
        >>> input = torch.randn((1,3,32,32), dtype=torch.float32)
        >>> net = torch.nn.Sequential(nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1),\
                                    nn.ReLU(),\
                                    SpatialPyramidPooling2d(num_level=2,pool_type='avg_pool'),\
                                    nn.Linear(32 * (1*1 + 2*2), 10))
        >>> output = net(input)
     """
     
    def __init__(self, num_level, pool_type='max_pool'):
        super(SpatialPyramidPooling2d, self).__init__()
        self.num_level = num_level
        self.pool_type = pool_type
    #卷积后的的大小为[1,64,29,29]
    def forward(self, x):
        N, C, H, W = x.size()
        for i in range(self.num_level):
            print(i)
            level = i + 1
            kernel_size = (ceil(H / level), ceil(W / level))
            stride = (ceil(H / level), ceil(W / level))
            padding = (floor((kernel_size[0] * level - H + 1) / 2), floor((kernel_size[1] * level - W + 1) / 2))
            #得到三个最大池化参数(29, 29),(29, 29),(0, 0)  (15, 15),(15, 15),(1, 1) (10, 10),(10, 10),(1, 1)
            #(29, 29),(29, 29),(0, 0) 为对整张图进行一次池化，得到1*64
            #(15, 15),(15, 15),(1, 1)， 按15*15池化，得到4*64
            #(10, 10),(10, 10),(1, 1)， 按10*10池化，得到9*64
            if self.pool_type == 'max_pool':
                tensor = (F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)).view(N, -1)
            else:
                tensor = (F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)).view(N, -1)
            
            if i == 0:
                res = tensor
            else:
                res = torch.cat((res, tensor), 1)#按列进行拼接1*64+4*64+9*64=14*64
        return res
    #类似于Java的toString()
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'num_level = ' + str(self.num_level) \
            + ', pool_type = ' + str(self.pool_type) + ')'
     
 
class SPPNet(nn.Module):
    def __init__(self, num_level=3, pool_type='max_pool'):
        super(SPPNet,self).__init__()
        self.num_level = num_level
        self.pool_type = pool_type
        #nn.Conv2d(3,64,3)图像3通道--》64通道 核为3*3  64*64-->62*62
        #nn.MaxPool2d(2) 核为2*2  62*62-->31*31
        #nn.Conv2d(64,64,3)特征图64通道--》64通道  核为3*3 31*31-->29*29
        self.feature = nn.Sequential(nn.Conv2d(3,64,3),\
                                    nn.ReLU(),\
                                    nn.MaxPool2d(2),\
                                    nn.Conv2d(64,64,3),\
                                    nn.ReLU())
        self.num_grid = self._cal_num_grids(num_level)
        self.spp_layer = SpatialPyramidPooling2d(num_level)
        print(self.spp_layer)
        #输入神经元个数为14*64，输出神经元个数为512
        #输入神经元个数为512，输出神经元个数为10
        self.linear = nn.Sequential(nn.Linear(self.num_grid * 64, 512),\
                                    nn.Linear(512, 10))
    ##按num_level个尺度，计算每个尺度大小分别为（1*1）+(2*2)+(3*3)=14
    def _cal_num_grids(self, level):
        count = 0
        for i in range(level):
            count += (i + 1) * (i + 1)
        return count

    def forward(self, x):
        x = self.feature(x)
        x = self.spp_layer(x)
        print(x.size())
        x = self.linear(x)
        return x
 
if __name__ == '__main__':
    a = torch.rand((1,3,64,64))
    net = SPPNet()
    output = net(a)
    print(output)