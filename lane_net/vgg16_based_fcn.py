import torch
import torch.nn as nn

class _vggStage(nn.Module):
    def __init__(self, intput_dim, output_dim, cnn_num, kernel_size, normal = True, max_pool=False):
        super(_vggStage, self).__init__()
        self.intput_dim = intput_dim
        self.output_dim = output_dim
        self.cnn_num = cnn_num
        self.kernel_size = kernel_size
        self.normal = normal
        self.max_pool = max_pool

        self.conv_1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size)
        self.conv_2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size)
        if max_pool:
            self.maxpooling = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if self.max_pool:
            x = self.maxpooling(x)
        x = self.conv_1(x)
        for _ in range(self.cnn_num):
            x = self.conv_2(x)
            if self.normal:
                x = nn.BatchNorm2d(x)
            x = nn.ReLU(x)
        return x

class _decoder_block(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size = 3, stride = 2, activate = True, use_bias=False):
        super(_decoder_block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self,kernel_size = kernel_size
        self.activate = activate
        self.use_bias = use_bias
        self.stride = stride
    def forward(self, x, y):
        x = nn.ConvTranspose2d(in_channels = self.input_dim, out_channels = self.output_dim,
                                kernel_size = self.kernel_size, stride = self.stride)
        x = nn.BatchNorm2d(x)
        x = nn.ReLU(x)
        res = x + y
        if self.activate:
            res = nn.BatchNorm2d(res)
            res = nn.ReLU(res)
        return res
class VGG16FCN_encoder(nn.Module):
    def __init__(self, encode_dims, cnn_nums, branch_dims, branch_cnn_nums):
        super(VGG16FCN_encoder, self).__init__()
        self.encode_dims = encode_dims
        self.branch_dims = branch_dims
        self.stages = []
        self.branchs = []
        self.stages.append(_vggStage(3, encode_dims[0], cnn_nums[0], kernel_size = 3))
        for i in range(1, len(encode_dims)):
            self.stages.append(_vggStage(encode_dims[i-1], encode_dims[i], cnn_nums[i], kernel_size=3, max_pool=True))
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for i in range(len(branch_dims)):
            self.branchs.append(_vggStage(encode_dims[-1], branch_dims[i], branch_cnn_nums[i], kernel_size=3))


    def forward(self, x):
        stages_ = []
        branchs_ =[]
        for stage in self.stages:
            x = stage(x)
            stages_.append(x)
        x = self.max_pool(x)
        for branch in self.branchs:
            x = branch(x)
            branchs_.append(x)

        return {"stages": stages_, "branchs": branchs_}

class VGG16FCN_decoder(nn.Module):
    def __init__(self, input_dim, ):
        super(VGG16FCN_decoder).__init__()

