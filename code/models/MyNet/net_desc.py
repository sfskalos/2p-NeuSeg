import torch
import torch.nn as nn
from .EfficientNet import EfficientNet
from .net_utils import TextureEncoder,DimensionalReduction,GradientInducedTransition,NeighborConnectionDecoder,ChannelAttention,SpatialAttention,EMA
from .net_utils import SimpleConcatFusion, SEFusion, CBAMFusion, SelfAttentionFusion, WeightSumFusion, NoFusion, IdentityFusion, Res2NetStyleFusion
import torch.nn.functional as F
from .Res2Net import res2net50_v1b_26w_4s

# 融合方法映射
FUSION_METHODS = {
    'GIT': GradientInducedTransition,
    'Concat': SimpleConcatFusion,
    'SE': SEFusion,
    'CBAM': CBAMFusion,
    'SelfAttention': SelfAttentionFusion,
    'WeightSum': WeightSumFusion,
    'NoFusion': NoFusion,
    'Identity': IdentityFusion,
    'Res2NetStyle': Res2NetStyleFusion,
}


class mynet(nn.Module):
    def __init__(
        self,
        channel=256,
        arc='EfficientNet-B4',
        M=[32,32,32],
        N=[8,16,32],
        nr_types=3,
        fusion_method='GIT',
        use_git=None,
        use_og=True,
    ):
        super(mynet, self).__init__()
        self.nr_types = nr_types
        self.use_og = use_og
        if use_git is None:
            self.use_git = fusion_method != 'NoFusion'
            effective_fusion_method = fusion_method
        else:
            self.use_git = use_git
            effective_fusion_method = 'GIT' if use_git else 'NoFusion'
        self.fusion_method = effective_fusion_method
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        in_channel_list = [512,1024,2048]
        self.texture_encoder = TextureEncoder() if (self.use_git or self.use_og) else None

        self.dr3 = DimensionalReduction(in_channel=in_channel_list[0],out_channel=channel)
        self.dr4 = DimensionalReduction(in_channel=in_channel_list[1],out_channel=channel)
        self.dr5 = DimensionalReduction(in_channel=in_channel_list[2], out_channel=channel)

        # 选择融合方法
        if effective_fusion_method == 'GIT':
            self.git = GradientInducedTransition(channel=channel, M=M, N=N)
        elif effective_fusion_method == 'Concat':
            self.git = SimpleConcatFusion(channel=channel)
        elif effective_fusion_method == 'SE':
            self.git = SEFusion(channel=channel)
        elif effective_fusion_method == 'CBAM':
            self.git = CBAMFusion(channel=channel)
        elif effective_fusion_method == 'SelfAttention':
            self.git = SelfAttentionFusion(channel=channel)
        elif effective_fusion_method == 'WeightSum':
            self.git = WeightSumFusion(channel=channel)
        elif effective_fusion_method == 'NoFusion':
            self.git = NoFusion(channel=channel)
        elif effective_fusion_method == 'Identity':
            self.git = IdentityFusion(channel=channel)
        elif effective_fusion_method == 'Res2NetStyle':
            self.git = Res2NetStyleFusion(channel=channel)
        else:
            raise ValueError(f"Unknown fusion method: {effective_fusion_method}")

        self.ncd_np = NeighborConnectionDecoder(channel=channel,out_channel=2)
        self.ncd_hv = NeighborConnectionDecoder(channel=channel,out_channel=2)
        self.ncd_tp = NeighborConnectionDecoder(channel=channel,out_channel=nr_types)
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self,x):
        r1, r2, r3, r4 = self.resnet(x)
        xr3 = self.dr3(r2)
        xr4 = self.dr4(r3)
        xr5 = self.dr5(r4)

        if self.texture_encoder is not None:
            xg, pg = self.texture_encoder(x)
        else:
            xg, pg = None, None

        zt3,zt4,zt5 = self.git(xr3,xr4,xr5,xg)
        dict = {}
        np_map = self.ncd_np(zt5, zt4, zt3)
        np_map = self.upsample(np_map)
        dict['np'] = np_map
        if self.use_og and pg is not None:
            dict['og'] = self.upsample(pg)
        else:
            dict['og'] = x.new_zeros((x.shape[0], 1, x.shape[2], x.shape[3]))
        hv = self.ncd_hv(zt5, zt4, zt3)
        hv = self.upsample(hv)
        tp = self.ncd_tp(zt5, zt4, zt3)
        tp = self.upsample(tp)
        dict['hv'] = hv
        dict['tp'] = tp

        return dict

def create_model(mode,**kwargs):
    return mynet(**kwargs)

# test
if __name__== '__main__':
    net = mynet(channel=512, arc='PVTv2-B2', M=[8, 8, 8], N=[4, 8, 16],nr_types =3).eval()
    inputs = torch.randn(1, 3, 384, 384)

    outs = net(inputs)
    print("Model output shapes:", {k: v.shape for k, v in outs.items()})
