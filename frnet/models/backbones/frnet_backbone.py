from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer,
                      build_norm_layer)
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.model import BaseModule
from torch import Tensor


class BasicBlock(BaseModule):

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 dilation: int = 1,
                 downsample: Optional[nn.Module] = None,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 act_cfg: ConfigType = dict(type='LeakyReLU'),
                 init_cfg: OptMultiConfig = None) -> None:
        super(BasicBlock, self).__init__(init_cfg)

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)
        self.relu = build_activation_layer(act_cfg)
        self.downsample = downsample

    @property
    def norm1(self) -> nn.Module:
        """nn.Module: Normalization layer after the first convolution layer."""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self) -> nn.Module:
        """nn.Module: Normalization layer after the second convolution layer.
        """
        return getattr(self, self.norm2_name)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


@MODELS.register_module()
class FRNetBackbone(BaseModule):

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3))
    }

    def __init__(self,
                 in_channels: int, #compressed 된 feature 크기와 같아야함, 이 코드에서는 16
                 point_in_channels: int, # 
                 output_shape: Sequence[int], # 논문에서는 (64,512) 
                 depth: int, # res18, res34 중 골라서 쓰는 듯, 이 코드에서는 34
                 stem_channels: int = 128, 
                 num_stages: int = 4,
                 out_channels: Sequence[int] = (128, 128, 128, 128),
                 strides: Sequence[int] = (1, 2, 2, 2),
                 dilations: Sequence[int] = (1, 1, 1, 1), # dilated convolution인듯? 세그멘테이션에서 효율 좋음
                 fuse_channels: Sequence[int] = (256, 128),
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'), # batch_norm_2d 
                 point_norm_cfg: ConfigType = dict(type='BN1d'), # batch_norm_1d
                 act_cfg: ConfigType = dict(type='LeakyReLU'), # activation function(이 코드에서는 HSwish(Hard Swish))
                 init_cfg: OptMultiConfig = None) -> None:
        super(FRNetBackbone, self).__init__(init_cfg)

        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for FRNetBackbone.') # depth 확인

        self.block, stage_blocks = self.arch_settings[depth] #(BasicBlock, (3,4,6,3))
        self.output_shape = output_shape # (64,512)
        self.ny = output_shape[0] # 64
        self.nx = output_shape[1] # 512
        assert len(stage_blocks) == len(out_channels) == len(strides) == len(
            dilations) == num_stages, \
            'The length of stage_blocks, out_channels, strides and ' \
            'dilations should be equal to num_stages.' # 블록, 채널, 스트라이드, dilations length 같은지 확인
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.point_norm_cfg = point_norm_cfg
        self.act_cfg = act_cfg
        self.stem = self._make_stem_layer(in_channels, stem_channels) # in_channels-> stem_channels//2 -> stem_channels 로 이어지는 convolution layer(bn, act 포함)
        self.point_stem = self._make_point_layer(point_in_channels, # point_in_channels-> stem_channels로 이어지는 Linear layer(bn,act 포함)
                                                 stem_channels)
        self.fusion_stem = self._make_fusion_layer(stem_channels * 2, # stem_channels * 2 -> stem_channels로 이어지는 conv(bn, act 포함) point의 feature와 frustum feature의 concat이기 때문에 x2인듯
                                                   stem_channels)

        inplanes = stem_channels #128
        self.res_layers = []
        self.point_fusion_layers = nn.ModuleList()
        self.pixel_fusion_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.strides = []
        overall_stride = 1
        for i, num_blocks in enumerate(stage_blocks):
            stride = strides[i]
            overall_stride = stride * overall_stride
            self.strides.append(overall_stride)
            dilation = dilations[i]
            planes = out_channels[i]
            res_layer = self.make_res_layer( # resnet의 Layer 쌓기 (3,4,6,3), 각각은 BasicBlock으로 이루어짐
                block=self.block,
                inplanes=inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
          
            self.point_fusion_layers.append(
                self._make_point_layer(inplanes + planes, planes)) # (256 -> 128 Channel) MLP
            self.pixel_fusion_layers.append(
                self._make_fusion_layer(planes * 2, planes)) #(256 -> 128 Channel) Convolution
            self.attention_layers.append(self._make_attention_layer(planes)) # (128 -> 128) convolution -> convolution -> activation(Sigmoid) : 0~1 사이의 score로 변환인듯?
            inplanes = planes
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        in_channels = stem_channels + sum(out_channels) # (128*5)
        self.fuse_layers = []
        self.point_fuse_layers = []
        for i, fuse_channel in enumerate(fuse_channels): # fuse_channels : (256, 128)
            fuse_layer = ConvModule(
                in_channels,
                fuse_channel,
                kernel_size=3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg) 
            point_fuse_layer = self._make_point_layer(in_channels, 
                                                      fuse_channel)
            in_channels = fuse_channel
            layer_name = f'fuse_layer{i + 1}'
            point_layer_name = f'point_fuse_layer{i + 1}'
            self.add_module(layer_name, fuse_layer) 
            self.add_module(point_layer_name, point_fuse_layer)
            self.fuse_layers.append(layer_name)
            self.point_fuse_layers.append(point_layer_name)

    def _make_stem_layer(self, in_channels: int,
                         out_channels: int) -> nn.Module:
        return nn.Sequential(
            build_conv_layer(
                self.conv_cfg,
                in_channels,
                out_channels // 2,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(self.norm_cfg, out_channels // 2)[1],
            build_activation_layer(self.act_cfg),
            build_conv_layer(
                self.conv_cfg,
                out_channels // 2,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(self.norm_cfg, out_channels)[1],
            build_activation_layer(self.act_cfg),
            build_conv_layer(
                self.conv_cfg,
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(self.norm_cfg, out_channels)[1],
            build_activation_layer(self.act_cfg))

    def _make_point_layer(self, in_channels: int,
                          out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            build_norm_layer(self.point_norm_cfg, out_channels)[1],
            nn.ReLU(inplace=True))

    def _make_fusion_layer(self, in_channels: int,
                           out_channels: int) -> nn.Module:
        return nn.Sequential(
            build_conv_layer(
                self.conv_cfg,
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(self.norm_cfg, out_channels)[1],
            build_activation_layer(self.act_cfg))

    def _make_attention_layer(self, channels: int) -> nn.Module:
        return nn.Sequential(
            build_conv_layer(
                self.conv_cfg,
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(self.norm_cfg, channels)[1],
            build_activation_layer(self.act_cfg),
            build_conv_layer(
                self.conv_cfg,
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(self.norm_cfg, channels)[1], nn.Sigmoid())

    def make_res_layer(
        self,
        block: nn.Module,
        inplanes: int,
        planes: int,
        num_blocks: int,
        stride: int,
        dilation: int,
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type='BN'),
        act_cfg: ConfigType = dict(type='LeakyReLU')
    ) -> nn.Module:
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(norm_cfg, planes)[1])

        layers = []
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))
        inplanes = planes
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=1,
                    dilation=dilation,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        return nn.Sequential(*layers)

    def forward(self, voxel_dict: dict) -> dict:

        point_feats = voxel_dict['point_feats'][-1] # 각 레이어마다 저장했기 때문에 마지막 레이어의 결과를 가져오는 것 같음 (N,256)
        voxel_feats = voxel_dict['voxel_feats'] # (M,C)
        voxel_coors = voxel_dict['voxel_coors'] # (M,3)
        pts_coors = voxel_dict['coors'] # (N, 3) -> (batch_index,y,x)
        batch_size = pts_coors[-1, 0].item() + 1 # 마지막 포인트의 batch index에 1을 더하면 batch size와 같음(index가 0부터 시작되었기 때문)

        x = self.frustum2pixel(voxel_feats, voxel_coors, batch_size, stride=1) # (batch size, channel=16, 64,512)
        x = self.stem(x) # (batch size, channel=128, 64,512)
        map_point_feats = self.pixel2point(x, pts_coors, stride=1) # (N, channel=128)
        fusion_point_feats = torch.cat((map_point_feats, point_feats), dim=1) #(N, 128+256)
        point_feats = self.point_stem(fusion_point_feats) # (N, 128) 차원 축소 
        stride_voxel_coors, frustum_feats = self.point2frustum( # frustum에 속한 point중에서의 feature를 local max한 후에 voxel마다 frustum feature로 저장
            point_feats, pts_coors, stride=1) #(M,3),(M,128) -> (batch,y,x) , (features)
        pixel_feats = self.frustum2pixel(
            frustum_feats, stride_voxel_coors, batch_size, stride=1)  # (batch, 128, 64, 512)
        fusion_pixel_feats = torch.cat((pixel_feats, x), dim=1) # (batch, 256, 64, 512)
        x = self.fusion_stem(fusion_pixel_feats)  # (batch, 128, 64, 512)

        outs = [x]  # [(batch, 128, 64, 512)]
        out_points = [point_feats] #[(N, 128)]
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)

            # frustum-to-point fusion
            map_point_feats = self.pixel2point(
                x, pts_coors, stride=self.strides[i]) #[(N, 128)]
            fusion_point_feats = torch.cat((map_point_feats, point_feats),
                                           dim=1) #[(N, 256)]
            point_feats = self.point_fusion_layers[i](fusion_point_feats) #[(N, 128)]

            # point-to-frustum fusion
            stride_voxel_coors, frustum_feats = self.point2frustum(
                point_feats, pts_coors, stride=self.strides[i])  # stride_voxel_coors: (M, 3), frustum_feats: (M, 128)
            pixel_feats = self.frustum2pixel( # (batch, 128, 64, 512)
                frustum_feats,
                stride_voxel_coors,
                batch_size,
                stride=self.strides[i])
            fusion_pixel_feats = torch.cat((pixel_feats, x), dim=1)
            fuse_out = self.pixel_fusion_layers[i](fusion_pixel_feats)
            # residual-attentive
            attention_map = self.attention_layers[i](fuse_out)
            x = fuse_out * attention_map + x
            outs.append(x)
            out_points.append(point_feats)

        for i in range(len(outs)):
            if outs[i].shape != outs[0].shape:
                outs[i] = F.interpolate(
                    outs[i],
                    size=outs[0].size()[2:],
                    mode='bilinear',
                    align_corners=True)

        outs[0] = torch.cat(outs, dim=1)
        out_points[0] = torch.cat(out_points, dim=1)

        for layer_name, point_layer_name in zip(self.fuse_layers,
                                                self.point_fuse_layers):
            fuse_layer = getattr(self, layer_name)
            outs[0] = fuse_layer(outs[0])
            point_fuse_layer = getattr(self, point_layer_name)
            out_points[0] = point_fuse_layer(out_points[0])

        voxel_dict['voxel_feats'] = outs
        voxel_dict['point_feats_backbone'] = out_points
        return voxel_dict

    def frustum2pixel(self,
                      frustum_features: Tensor,
                      coors: Tensor,
                      batch_size: int,
                      stride: int = 1) -> Tensor:
        nx = self.nx // stride
        ny = self.ny // stride
        pixel_features = torch.zeros(
            (batch_size, ny, nx, frustum_features.shape[-1]), # (batch size, ny, nx, voxel의 feature) 
            dtype=frustum_features.dtype,
            device=frustum_features.device)
        pixel_features[coors[:, 0], coors[:, 1], coors[:,
                                                       2]] = frustum_features
        pixel_features = pixel_features.permute(0, 3, 1, 2).contiguous() # (batch size, voxel feature, ny, nx) 순으로 재조정
        return pixel_features

    def pixel2point(self,
                    pixel_features: Tensor,
                    coors: Tensor,
                    stride: int = 1) -> Tensor:
        pixel_features = pixel_features.permute(0, 2, 3, 1).contiguous() #(batch size, ny, nx, feature)
        point_feats = pixel_features[coors[:, 0], coors[:, 1] // stride, #(batch size, N, feature)
                                     coors[:, 2] // stride]
        return point_feats

    def point2frustum(self,
                      point_features: Tensor,
                      pts_coors: Tensor,
                      stride: int = 1) -> Tuple[Tensor, Tensor]:
        coors = pts_coors.clone()
        coors[:, 1] = pts_coors[:, 1] // stride
        coors[:, 2] = pts_coors[:, 2] // stride
        voxel_coors, inverse_map = torch.unique( # voxel_coors : (batch size, H,W)
            coors, return_inverse=True, dim=0) 
        frustum_features = torch_scatter.scatter_max( (batch size, 128)
            point_features, inverse_map, dim=0)[0]
        return voxel_coors, frustum_features
