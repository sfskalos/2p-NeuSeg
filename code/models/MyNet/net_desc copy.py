import torch
import torch.nn as nn
from EfficientNet import EfficientNet
from net_utils import TextureEncoder,DimensionalReduction,GradientInducedTransition,NeighborConnectionDecoder,ChannelAttention,SpatialAttention
import torch.nn.functional as F

class mynet(nn.Module):
    def __init__(self,channel=64,arc='EfficientNet-B4',M=[8,8,8],N=[4,8,16],nr_types=None):
        super(mynet, self).__init__()
        self.nr_types = nr_types
        self.context_encoder = EfficientNet.from_pretrained('efficientnet-b4')
        in_channel_list = [56, 160, 448]
        self.texture_encoder = TextureEncoder()

        self.dr3 = DimensionalReduction(in_channel=in_channel_list[0],out_channel=channel)
        self.dr4 = DimensionalReduction(in_channel=in_channel_list[1],out_channel=channel)
        self.dr5 = DimensionalReduction(in_channel=in_channel_list[2], out_channel=channel)

        self.git = GradientInducedTransition(channel=channel, M=M, N=N)
        # self.ncd_np = NeighborConnectionDecoder(channel=channel,out_channel=2)
        self.ncd_hv = NeighborConnectionDecoder(channel=channel,out_channel=2)
        self.ncd_tp = NeighborConnectionDecoder(channel=channel,out_channel=nr_types)
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.ca = ChannelAttention(channel, 8)  # 通道注意力实例
        self.sa = SpatialAttention(3)

    def cbma(self,x):
        x= x*self.ca(x)
        x = x*self.sa(x)
        # x=x*
        return x
    
    def forward(self,x):
        endpoints = self.context_encoder.extract_endpoints(x)
        x3 = endpoints['reduction_3']
        x4 = endpoints['reduction_4']
        x5 = endpoints['reduction_5']
        # dr统一维度[1,6,]
        xr3 = self.dr3(x3)
        xr4 = self.dr4(x4)
        xr5 = self.dr5(x5)

        xg, pg = self.texture_encoder(x)
        zt3,zt4,zt5 = self.git(xr3,xr4,xr5,xg)

        # zt3 = self.cbma(zt3)
        # zt4 = self.cbma(zt4)
        # zt5 = self.cbma(zt5)


        dict = {}
        # np branch
        # np = self.ncd_np(zt5,zt4,zt3)
        # dict['np'] = self.upsample(np)
        # pg = self.cbma(pg)
        dict['og'] = self.upsample(pg)

        
        # hv branch
        hv = self.ncd_hv(zt5, zt4, zt3)
        
        hv = self.upsample(hv)
        
        # tp branch
        tp = self.ncd_tp(zt5, zt4, zt3)
        
        tp = self.upsample(tp)
        dict['confidence'] = F.softmax(tp, dim=1)
        uncertainty = 1-dict['confidence']

        tp = tp*uncertainty
        dict['hv'] = hv
        dict['tp'] = tp

        
        return dict

def create_model(mode,**kwargs):
    return mynet(**kwargs)

# test
if __name__== '__main__':
    
    net = mynet(channel=64, arc='PVTv2-B2', M=[8, 8, 8], N=[4, 8, 16],nr_types =3).eval()
    inputs = torch.randn(1, 3, 384, 384)

    outs = net(inputs)
# a=1