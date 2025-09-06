import torch
import torch.nn as nn
from ..utils.conver_box import cell_xywh2xyxy
from ..utils.helpers import clip_boxes


class ConvBA(nn.Module):
    def __init__(
            self, in_channels, out_channels,
            kernel_size, stride=1, padding=0,
            bn=True, act="leakyrelu"
        ):
        super().__init__()
        self.bn = bn
        self.act = act
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=(not self.bn)
        )

        if self.bn:
            self.batchnorm = nn.BatchNorm2d(out_channels)

        self.fn_act = nn.ModuleDict({
            "relu": nn.ReLU(inplace=True),
            "leakyrelu": nn.LeakyReLU(inplace=True),
            "gelu": nn.GELU()
        })

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.batchnorm(x)
        out = self.fn_act[self.act](x)
        return out
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, act="leakyrelu"):
        super().__init__()
        self.conv1x1 = ConvBA(in_channels, out_channels//2, (1,1), bn=bn, act=act)
        self.conv3x3 = ConvBA(out_channels//2, out_channels, (3,3), padding=1, bn=bn, act=act)

    def forward(self, x):
        x = self.conv1x1(x)
        out = self.conv3x3(x)
        return out


class YoloV1Backbone(nn.Module):
    def __init__(self, bn=True, act="leakyrelu"):
        super().__init__()
        self.conv1 = ConvBA(3, 64, (7,7), stride=2, padding=3, bn=bn, act=act)
        self.conv2 = ConvBA(64, 192, (3,3), padding=1, bn=bn, act=act)
        self.conv3 = ConvBlock(192, 256, bn=bn, act=act)
        self.conv4 = ConvBlock(256, 512, bn=bn, act=act)
        self.conv5 = ConvBlock(512, 512, bn=bn, act=act)
        self.conv6 = ConvBlock(512, 512, bn=bn, act=act)
        self.conv7 = ConvBlock(512, 512, bn=bn, act=act)
        self.conv8 = ConvBlock(512, 512, bn=bn, act=act)
        self.conv9 = ConvBlock(512, 1024, bn=bn, act=act)
        self.conv10 = ConvBlock(1024, 1024, bn=bn, act=act)
        self.conv11 = ConvBlock(1024, 1024, bn=bn, act=act)
        self.max_pool = nn.MaxPool2d((2,2), stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.max_pool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.max_pool(x)
        x = self.conv10(x)
        out = self.conv11(x)
        return out


class YoloV1(nn.Module):
    def __init__(
            self, S, B, num_cls,
            backbone=None, bn=True,
            use_layer=None,
            act="leakyrelu", dropout=0.5
        ):
        super().__init__()
        self.S = S
        self.B = B
        self.num_cls = num_cls
        self.use_layer = use_layer
        self.act = act
        self.fn_act = nn.ModuleDict({
            "relu": nn.ReLU(),
            "leakyrelu": nn.LeakyReLU(),
            "gelu": nn.GELU()
        })

        if backbone is None:
            self.backbone = YoloV1Backbone(bn=bn, act=self.act)
            out_channels = 1024
        else:
            self.backbone = backbone
            for layer in backbone.modules():
                if isinstance(layer, nn.Conv2d):
                    out_channels = layer.out_channels

        self.neck = nn.Sequential(
            ConvBA(out_channels, 1024, (3,3), padding=1, bn=bn, act=self.act),
            ConvBA(1024, 1024, (3,3), stride=2, padding=1, bn=bn, act=self.act),
            ConvBA(1024, 1024, (3,3), padding=1, bn=bn, act=self.act),
            ConvBA(1024, 1024, (3,3), padding=1, bn=bn, act=self.act)
        )

        if self.use_layer == "conv1x1":
            self.head = nn.Conv2d(1024, 5 * self.B + self.num_cls, (1,1))
        else:
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.S*self.S*1024, 4096),
                self.fn_act[self.act],
                nn.Dropout(p=dropout),
                nn.Linear(4096, self.S*self.S*(5*self.B+self.num_cls))
            )


    def forward(self, x):
        batch_size = x.shape[0]

        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)

        if self.use_layer == "conv1x1":
            out = x.permute(0, 2, 3, 1)
        else: 
            out = x.reshape(batch_size, self.S, self.S, -1)

        bc = torch.nn.functional.sigmoid(out[..., :5*self.B])
        out = torch.cat([bc, out[..., 5*self.B:]], dim=-1)
        return out if self.training else (out, self.convert_bnboxes(out))
    

    def convert_bnboxes(self, out):
        c_out = out.clone()
        batch_size = c_out.shape[0]

        bnboxes_cord = c_out[..., :5*self.B]
        # (batch, 7, 7, 2, 5)
        bnboxes_cord = bnboxes_cord.reshape(batch_size, self.S, self.S, self.B, -1)
        bnboxes_cord[..., 2:4] = bnboxes_cord[..., 2:4].square()
        # (batch, 7, 7, 2, 4)
        boxes_xyxy = cell_xywh2xyxy(bnboxes_cord[..., :4])
        # (batch, 7, 7, 2, 5)
        clip_boxes(boxes_xyxy)
        bnboxes_cord[..., :4] =  boxes_xyxy
        # (batch, 7, 7, 20)
        bnboxes_cls = c_out[..., 5*self.B:]
        # (batch, 7, 7, 1, 20)
        bnboxes_cls = bnboxes_cls.reshape(batch_size, self.S, self.S, 1, -1)
        # (batch, 7, 7, 2, 20)
        bnboxes_cls = bnboxes_cls.repeat(1, 1, 1, self.B, 1)
        # (batch, 98, 25)
        bnboxes = torch.cat([bnboxes_cord, bnboxes_cls], dim=-1).reshape(batch_size, self.S*self.S*self.B, -1)

        return bnboxes
    