import torch
from torch import nn


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, padding=0):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=padding, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=padding, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch, padding=1):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=padding, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
        )

    def forward(self, x):
        x = self.up(x)
        return x

class Unet_prototype(nn.Module):
    def __init__(self, in_ch=1, out_ch=4, proj_dim=128):
        super(Unet_prototype, self).__init__()

        filters = [64, 128, 256, 512, 1024]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(in_ch, filters[0], padding=1)
        self.Conv2 = conv_block(filters[0], filters[1], padding=1)
        self.Conv3 = conv_block(filters[1], filters[2], padding=1)
        self.Conv4 = conv_block(filters[2], filters[3], padding=1)
        self.Conv5 = conv_block(filters[3], filters[4], padding=1)

        self.Up4 = up_conv(filters[4], 4, padding=1)
        self.Up_conv4 = conv_block(516, filters[3], padding=1)
        self.Up3 = up_conv(filters[3], 4, padding=1)
        self.Up_conv3 = conv_block(260, filters[2], padding=1)
        self.Up2 = up_conv(filters[2], filters[1], padding=1)
        self.Up_conv2 = conv_block(filters[2], filters[1], padding=1)
        self.Up1 = up_conv(filters[1], filters[0], padding=1)
        self.Up_conv1 = conv_block(filters[1], filters[0], padding=1)
        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.Norm = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01)
        self.active = torch.nn.Softmax(dim=1)

        self.deep_seg_head_list = nn.ModuleList([
            nn.Sequential(nn.Conv2d(filters[4 - abs(i - 4)], out_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
            torch.nn.Softmax(dim=1))
            for i in range(9)
        ])

        self.proj_head_list = nn.ModuleList([
            nn.Conv2d(filters[4 - abs(i - 4)], proj_dim, kernel_size=1, stride=1, padding=0)
            for i in range(9)
        ])

    def forward(self, tensor_list):
        x1 = tensor_list

        e1 = self.Conv1(x1) # B x 64 x 224 x 224

        e2 = self.Maxpool1(e1) # B x 64 x 112 x 112
        e2 = self.Conv2(e2) # B x 128 x 112 x 112

        e3 = self.Maxpool2(e2) # B x 128 x 56 x 56
        e3 = self.Conv3(e3) # B x 256 x 56 x 56

        e4 = self.Maxpool3(e3) # B x 256 x 28 x 28
        e4 = self.Conv4(e4) # B x 512 x 28 x 28

        # ------------------------------------------
        # Construct prototype
        # ------------------------------------------
        e5 = self.Maxpool4(e4) # B x 512 x 14 x 14
        e5 = self.Conv5(e5) # B x 1024 x 14 x 14

        d4 = self.Up4(e5) # B x 4 x 28 x 28
        d4 = torch.cat((d4, e4), dim=1) # B x 516 x 28 x 28
        d4 = self.Up_conv4(d4) # B x 512 x 28 x 28

        d3 = self.Up3(d4) # B x 4 x 56 x 56
        d3 = torch.cat((d3, e3), dim=1) # B x 260 x 56 x 56
        d3 = self.Up_conv3(d3) # B x 256 x 56 x 56

        d2 = self.Up2(d3) # B x 128 x 112 x 112
        d2 = torch.cat((d2, e2), dim=1) # B x 256 x 112 x 112
        d2 = self.Up_conv2(d2) # B x 128 x 112 x 112

        d1 = self.Up1(d2) # B x 64 x 224 x 224
        d1 = torch.cat((d1, e1), dim=1) # B x 128 x 224 x 224
        d1 = self.Up_conv1(d1) # B x 64 x 224 x 224

        d0 = self.Conv(d1) # B x 4 x 224 x 224
        norm_out = self.Norm(d0) # B x 4 x 224 x 224
        out = self.active(norm_out) # B x 4 x 224 x 224

        self.feature_list = [e1, e2, e3, e4, e5, d4, d3, d2, d1]
        self.deep_seg_list = []
        for i in range(len(self.feature_list)):
            self.deep_seg_list.append(self.deep_seg_head_list[i](self.feature_list[i]))
        self.proj_list = []
        for i in range(len(self.feature_list)):
            self.proj_list.append(self.proj_head_list[i](self.feature_list[i]))
        return out

    def get_feature_list(self):
        return self.feature_list
    def get_deep_seg_list(self):
        return self.deep_seg_list
    def get_proj_list(self):
        return self.proj_list


if __name__ == '__main__':
    model = Unet_prototype()
    x = torch.rand(4, 1, 224, 224)
    y = model(x)
    print(y.shape)