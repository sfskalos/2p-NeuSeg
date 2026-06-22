import torch.nn as nn
import torch
import torch.nn.functional as F

class ConvBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvBR, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class TextureEncoder(nn.Module):
    def __init__(self):
        super(TextureEncoder, self).__init__()
        self.conv1 = ConvBR(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = ConvBR(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = ConvBR(64, 32, kernel_size=3, stride=2, padding=1)
        self.conv_out = ConvBR(32, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        feat = self.conv1(x)
        #[N,64,192,192]
        feat = self.conv2(feat)
        #[N,64,96,96]
        xg = self.conv3(feat)
        #[N,32,48,48]
        pg = self.conv_out(xg)
        return xg, pg

class DimensionalReduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DimensionalReduction, self).__init__()
        self.reduce = nn.Sequential(
            ConvBR(in_channel, out_channel, 3, padding=1),
            ConvBR(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)

class SoftGroupingStrategy(nn.Module):
    def __init__(self, in_channel, out_channel, N):
        super(SoftGroupingStrategy, self).__init__()
        self.g_conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, groups=N[0], bias=False)
        self.g_conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=1, groups=N[1], bias=False)
        self.g_conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, groups=N[2], bias=False)

    def forward(self, q):
        return self.g_conv1(q) + self.g_conv2(q) + self.g_conv3(q)

class GradientInducedTransition(nn.Module):
    def __init__(self, channel, M, N):
        super(GradientInducedTransition, self).__init__()
        self.M = M

        self.downsample2 = nn.Upsample(scale_factor=1 / 2, mode='bilinear', align_corners=True)#实际上是下采样
        self.downsample4 = nn.Upsample(scale_factor=1 / 4, mode='bilinear', align_corners=True)

        self.sgs3 = SoftGroupingStrategy(channel + 32, channel, N=N)
        self.sgs4 = SoftGroupingStrategy(channel + 32, channel, N=N)
        self.sgs5 = SoftGroupingStrategy(channel + 32, channel, N=N)

    def forward(self, xr3, xr4, xr5, xg):#xr345来自res2net，xgtextureenc
        # transmit the gradient cues into the context embeddings
        q3 = self.gradient_induced_feature_grouping(xr3, xg, M=self.M[0])
        q4 = self.gradient_induced_feature_grouping(xr4, self.downsample2(xg), M=self.M[1])
        q5 = self.gradient_induced_feature_grouping(xr5, self.downsample4(xg), M=self.M[2])

        # attention residual learning
        zt3 = xr3 + self.sgs3(q3)
        zt4 = xr4 + self.sgs4(q4)
        zt5 = xr5 + self.sgs5(q5)

        return zt3, zt4, zt5

    def gradient_induced_feature_grouping(self, xr, xg, M):
        if not M in [1, 2, 4, 8, 16, 32]:
            raise ValueError("Invalid Group Number!: must be one of [1, 2, 4, 8, 16, 32]")

        if M == 1:
            return torch.cat((xr, xg), 1)

        xr_g = torch.chunk(xr, M, dim=1)#将xr切分为dim'维M份
        xg_g = torch.chunk(xg, M, dim=1)
        foo = list()
        for i in range(M):
            foo.extend([xr_g[i], xg_g[i]])

        return torch.cat(foo, 1)

class NeighborConnectionDecoder(nn.Module):
    def __init__(self, channel,out_channel=1):
        super(NeighborConnectionDecoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = ConvBR(channel, channel, 3, padding=1)
        self.conv_upsample2 = ConvBR(channel, channel, 3, padding=1)
        self.conv_upsample3 = ConvBR(channel, channel, 3, padding=1)
        self.conv_upsample4 = ConvBR(channel, channel, 3, padding=1)
        self.conv_upsample5 = ConvBR(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = ConvBR(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = ConvBR(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = ConvBR(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel,out_channel, 1)

    def forward(self, zt5, zt4, zt3):
        zt5_1 = zt5
        zt4_1 = self.conv_upsample1(self.upsample(zt5)) * zt4
        zt3_1 = self.conv_upsample2(self.upsample(zt4_1)) * self.conv_upsample3(self.upsample(zt4)) * zt3

        zt4_2 = torch.cat((zt4_1, self.conv_upsample4(self.upsample(zt5_1))), 1)
        zt4_2 = self.conv_concat2(zt4_2)

        zt3_2 = torch.cat((zt3_1, self.conv_upsample5(self.upsample(zt4_2))), 1)
        zt3_2 = self.conv_concat3(zt3_2)

        pc = self.conv4(zt3_2)
        pc = self.conv5(pc)

        return pc


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 自适应最大池化

        # 两个卷积层用于从池化后的特征中学习注意力权重
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)  # 第一个卷积层，降维
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)  # 第二个卷积层，升维
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):
        # 对平均池化的特征进行处理
        avg_out = self.avg_pool(x)
        avg_out = self.fc1(avg_out)
        avg_out = self.relu1(avg_out)
        avg_out = self.fc2(avg_out)
        # 对最大池化的特征进行处理
        max_out = self.max_pool(x)
        max_out = self.fc1(max_out)
        max_out = self.relu1(max_out)
        max_out = self.fc2(max_out)
    
        out = avg_out + max_out  # 将两种池化的特征加权和作为输出
        return self.sigmoid(out)  # 使用sigmoid激活函数计算注意力权重
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'  # 核心大小只能是3或7
        padding = 3 if kernel_size == 7 else 1  # 根据核心大小设置填充

        # 卷积层用于从连接的平均池化和最大池化特征图中学习空间注意力权重
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 对输入特征图执行平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 对输入特征图执行最大池化
        x = torch.cat([avg_out, max_out], dim=1)  # 将两种池化的特征图连接起来
        x = self.conv1(x)  # 通过卷积层处理连接后的特征图
        return self.sigmoid(x)  # 使用sigmoid激活函数计算注意力权重


class EMA(nn.Module):
    def __init__(self, channels, factor=4):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

# ==========================================
# GIT替代模块 - 消融实验
# ==========================================

class SimpleConcatFusion(nn.Module):
    """方法1：简单拼接 + 卷积融合"""
    def __init__(self, channel):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            ConvBR(channel + 32, channel, 3, padding=1),
            ConvBR(channel, channel, 3, padding=1)
        )

    def forward(self, xr3, xr4, xr5, xg):
        zt3 = self.fusion_conv(torch.cat([xr3, xg], dim=1))
        zt4 = self.fusion_conv(torch.cat([xr4, F.interpolate(xg, size=xr4.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        zt5 = self.fusion_conv(torch.cat([xr5, F.interpolate(xg, size=xr5.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        return zt3, zt4, zt5


class SEFusion(nn.Module):
    """方法2：SE注意力机制融合"""
    def __init__(self, channel):
        super().__init__()
        self.channel = channel
        self.se_block3 = self._make_se_block(channel + 32)
        self.se_block4 = self._make_se_block(channel + 32)
        self.se_block5 = self._make_se_block(channel + 32)
        self.fusion_conv3 = ConvBR(channel + 32, channel, 1)
        self.fusion_conv4 = ConvBR(channel + 32, channel, 1)
        self.fusion_conv5 = ConvBR(channel + 32, channel, 1)

    def _make_se_block(self, channels):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, xr3, xr4, xr5, xg):
        xg4 = F.interpolate(xg, size=xr4.shape[2:], mode='bilinear', align_corners=True)
        xg5 = F.interpolate(xg, size=xr5.shape[2:], mode='bilinear', align_corners=True)

        cat3 = torch.cat([xr3, xg], dim=1)
        cat4 = torch.cat([xr4, xg4], dim=1)
        cat5 = torch.cat([xr5, xg5], dim=1)

        weight3 = self.se_block3(cat3)
        weight4 = self.se_block4(cat4)
        weight5 = self.se_block5(cat5)

        zt3 = xr3 + self.fusion_conv3(cat3 * weight3)
        zt4 = xr4 + self.fusion_conv4(cat4 * weight4)
        zt5 = xr5 + self.fusion_conv5(cat5 * weight5)
        return zt3, zt4, zt5


class CBAMFusion(nn.Module):
    """方法3：CBAM注意力融合"""
    def __init__(self, channel):
        super().__init__()
        self.channel = channel
        self.ca3 = ChannelAttention(channel + 32)
        self.ca4 = ChannelAttention(channel + 32)
        self.ca5 = ChannelAttention(channel + 32)
        self.sa3 = SpatialAttention()
        self.sa4 = SpatialAttention()
        self.sa5 = SpatialAttention()
        self.fusion_conv3 = ConvBR(channel + 32, channel, 1)
        self.fusion_conv4 = ConvBR(channel + 32, channel, 1)
        self.fusion_conv5 = ConvBR(channel + 32, channel, 1)

    def forward(self, xr3, xr4, xr5, xg):
        xg4 = F.interpolate(xg, size=xr4.shape[2:], mode='bilinear', align_corners=True)
        xg5 = F.interpolate(xg, size=xr5.shape[2:], mode='bilinear', align_corners=True)

        cat3 = torch.cat([xr3, xg], dim=1)
        cat4 = torch.cat([xr4, xg4], dim=1)
        cat5 = torch.cat([xr5, xg5], dim=1)

        ca3 = self.ca3(cat3)
        ca4 = self.ca4(cat4)
        ca5 = self.ca5(cat5)

        sa3 = self.sa3(cat3)
        sa4 = self.sa4(cat4)
        sa5 = self.sa5(cat5)

        cat3_att = cat3 * ca3 * sa3
        cat4_att = cat4 * ca4 * sa4
        cat5_att = cat5 * ca5 * sa5

        zt3 = xr3 + self.fusion_conv3(cat3_att)
        zt4 = xr4 + self.fusion_conv4(cat4_att)
        zt5 = xr5 + self.fusion_conv5(cat5_att)
        return zt3, zt4, zt5


class SelfAttentionFusion(nn.Module):
    """方法4：自注意力融合"""
    def __init__(self, channel):
        super().__init__()
        self.channel = channel
        self.query = nn.Conv2d(channel + 32, channel // 4, 1)
        self.key = nn.Conv2d(channel + 32, channel // 4, 1)
        self.value = nn.Conv2d(channel + 32, channel, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.fusion_conv = ConvBR(channel, channel, 1)

    def forward(self, xr3, xr4, xr5, xg):
        xg4 = F.interpolate(xg, size=xr4.shape[2:], mode='bilinear', align_corners=True)
        xg5 = F.interpolate(xg, size=xr5.shape[2:], mode='bilinear', align_corners=True)

        cat3 = torch.cat([xr3, xg], dim=1)
        cat4 = torch.cat([xr4, xg4], dim=1)
        cat5 = torch.cat([xr5, xg5], dim=1)

        b, c, h, w = cat3.shape
        q3 = self.query(cat3).flatten(2).permute(0, 2, 1)
        k3 = self.key(cat3).flatten(2)
        v3 = self.value(cat3).flatten(2).permute(0, 2, 1)
        attn3 = torch.bmm(q3, k3) / (c ** 0.5)
        attn3 = torch.softmax(attn3, dim=-1)
        out3 = torch.bmm(attn3, v3).permute(0, 2, 1).reshape(b, -1, h, w)

        b, c, h, w = cat4.shape
        q4 = self.query(cat4).flatten(2).permute(0, 2, 1)
        k4 = self.key(cat4).flatten(2)
        v4 = self.value(cat4).flatten(2).permute(0, 2, 1)
        attn4 = torch.bmm(q4, k4) / (c ** 0.5)
        attn4 = torch.softmax(attn4, dim=-1)
        out4 = torch.bmm(attn4, v4).permute(0, 2, 1).reshape(b, -1, h, w)

        b, c, h, w = cat5.shape
        q5 = self.query(cat5).flatten(2).permute(0, 2, 1)
        k5 = self.key(cat5).flatten(2)
        v5 = self.value(cat5).flatten(2).permute(0, 2, 1)
        attn5 = torch.bmm(q5, k5) / (c ** 0.5)
        attn5 = torch.softmax(attn5, dim=-1)
        out5 = torch.bmm(attn5, v5).permute(0, 2, 1).reshape(b, -1, h, w)

        zt3 = xr3 + self.gamma * self.fusion_conv(out3)
        zt4 = xr4 + self.gamma * self.fusion_conv(out4)
        zt5 = xr5 + self.gamma * self.fusion_conv(out5)
        return zt3, zt4, zt5


class WeightSumFusion(nn.Module):
    """方法5：可学习的加权求和"""
    def __init__(self, channel):
        super().__init__()
        self.channel = channel
        self.weight_xr = nn.Parameter(torch.ones(1))
        self.weight_xg = nn.Parameter(torch.ones(1))
        self.fusion_conv = nn.Sequential(
            ConvBR(channel + 32, channel, 3, padding=1),
            ConvBR(channel, channel, 3, padding=1)
        )

    def forward(self, xr3, xr4, xr5, xg):
        xg4 = F.interpolate(xg, size=xr4.shape[2:], mode='bilinear', align_corners=True)
        xg5 = F.interpolate(xg, size=xr5.shape[2:], mode='bilinear', align_corners=True)

        weight_sum3 = self.weight_xr * xr3 + self.weight_xg * xg
        weight_sum4 = self.weight_xr * xr4 + self.weight_xg * xg4
        weight_sum5 = self.weight_xr * xr5 + self.weight_xg * xg5

        zt3 = xr3 + self.fusion_conv(weight_sum3)
        zt4 = xr4 + self.fusion_conv(weight_sum4)
        zt5 = xr5 + self.fusion_conv(weight_sum5)
        return zt3, zt4, zt5


class NoFusion(nn.Module):
    """方法6：不进行融合（直接使用上下文特征）"""
    def __init__(self, channel):
        super().__init__()

    def forward(self, xr3, xr4, xr5, xg):
        return xr3, xr4, xr5


class IdentityFusion(nn.Module):
    """方法7：恒等映射（保留GIT结构但不做任何操作）"""
    def __init__(self, channel):
        super().__init__()

    def forward(self, xr3, xr4, xr5, xg):
        return xr3, xr4, xr5


class Res2NetStyleFusion(nn.Module):
    """方法8：Res2Net风格的多尺度特征融合"""
    def __init__(self, channel):
        super().__init__()
        self.channel = channel
        self.scale_conv = nn.ModuleList([
            ConvBR(32, channel, 3, padding=1) for _ in range(3)
        ])

    def forward(self, xr3, xr4, xr5, xg):
        xg4 = F.interpolate(xg, size=xr4.shape[2:], mode='bilinear', align_corners=True)
        xg5 = F.interpolate(xg, size=xr5.shape[2:], mode='bilinear', align_corners=True)

        xg_processed3 = self.scale_conv[0](xg)
        xg_processed4 = self.scale_conv[1](xg4)
        xg_processed5 = self.scale_conv[2](xg5)

        zt3 = xr3 + xg_processed3
        zt4 = xr4 + xg_processed4
        zt5 = xr5 + xg_processed5
        return zt3, zt4, zt5