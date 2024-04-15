from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch_scatter
from mmcv.cnn import build_norm_layer
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType


@MODELS.register_module()
class FrustumFeatureEncoder(nn.Module):
    """Frustum Feature Encoder.

    Args:
        in_channels (int): Number of input features, either x, y, z or
            x, y, z, r. Defaults to 4.
        feat_channels (Sequence[int]): Number of features in each of the N
            FFELayers. Defaults to [].
        with_distance (bool): Whether to include Euclidean distance to points.
            Defaults to False.
        with_cluster_center (bool): Whether to include cluster center.
            Defaults to False.
        norm_cfg (dict or :obj:`ConfigDict`): Config dict of normalization
            layers. Defaults to dict(type='BN1d', eps=1e-5, momentum=0.1).
        with_pre_norm (bool): Whether to use the norm layer before input ffe
            layer. Defaults to False.
        feat_compression (int, optional): The frustum feature compression
            channels. Defaults to None.
    """

    def __init__(self,
                 in_channels: int = 4,
                 feat_channels: Sequence[int] = [],
                 with_distance: bool = False, # 유클리디안 distance
                 with_cluster_center: bool = False, # 각 cell에 포함된 포인트끼리의 평균에서의 상대적 거리
                 norm_cfg: ConfigType = dict(
                     type='BN1d', eps=1e-5, momentum=0.1), # batch_norm
                 with_pre_norm: bool = False, # 네트워크에 넣기전에 batch_norm 할것인지
                 feat_compression: Optional[int] = None) -> None: # 마지막 레이어에서 압축할 feature의 차원
        super(FrustumFeatureEncoder, self).__init__()
        assert len(feat_channels) > 0

        if with_distance:
            in_channels += 1
        if with_cluster_center:
            in_channels += 3
        self.in_channels = in_channels
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center

        feat_channels = [self.in_channels] + list(feat_channels) # [4, feat1, feat2,...]
        if with_pre_norm:
            self.pre_norm = build_norm_layer(norm_cfg, self.in_channels)[1]
        else:
            self.pre_norm = None
    ###########################Layer 생성(단순 MLP)###############################
        ffe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            norm_layer = build_norm_layer(norm_cfg, out_filters)[1]
            if i == len(feat_channels) - 2:
                ffe_layers.append(nn.Linear(in_filters, out_filters))
            else:
                ffe_layers.append(
                    nn.Sequential(
                        nn.Linear(in_filters, out_filters, bias=False),
                        norm_layer, nn.ReLU(inplace=True)))
        self.ffe_layers = nn.ModuleList(ffe_layers)
        self.compression_layers = None
        if feat_compression is not None:
            self.compression_layers = nn.Sequential(
                nn.Linear(feat_channels[-1], feat_compression),
                nn.ReLU(inplace=True))

    #############################################################################

    
    def forward(self, voxel_dict: dict) -> dict:
        features = voxel_dict['voxels']
        coors = voxel_dict['coors']
        features_ls = [features]

        voxel_coors, inverse_map = torch.unique(
            coors, return_inverse=True, dim=0) # 겹치지 않는 voxel_coors, 겹치는 것끼리 같은 인덱스 ex) [0,1,0,0,2,...]

        if self._with_distance:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist) #[(N,4),(N,1)]

        # Find distance of x, y, and z from frustum center
        if self._with_cluster_center:
            voxel_mean = torch_scatter.scatter_mean(
                features, inverse_map, dim=0)
            points_mean = voxel_mean[inverse_map]
            f_cluster = features[:, :3] - points_mean[:, :3] # 각 cell에 포함된 포인트끼리의 평균과 상대적인 거리
            features_ls.append(f_cluster) #[(N,4),(N,1),(N,3)]

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1) # [(N,8)]
        if self.pre_norm is not None:
            features = self.pre_norm(features)

        point_feats = []
        for ffe in self.ffe_layers:
            features = ffe(features)
            point_feats.append(features) # 각 레이어를 거쳤을때 feature 저장
        voxel_feats = torch_scatter.scatter_max(
            features, inverse_map, dim=0)[0] # 같은 cell에 포함된 포인트 중에서 feature가 가장 큰것을 voxel_feats에 할당(논문의 MaxPooling 파트) ex) 한개의 셀에 [1,0,0], [0.5,0.2,0.1], [0,0,0.6] 인 경우 이 셀은 [1,0.2,0.6]을 할당받게 된다.

        if self.compression_layers is not None:
            voxel_feats = self.compression_layers(voxel_feats) # maxpooling 이후 compression

        voxel_dict['voxel_feats'] = voxel_feats # 각 셀의 feature
        voxel_dict['voxel_coors'] = voxel_coors # 각 셀의 y,x
        voxel_dict['point_feats'] = point_feats # 각 레이어를 거쳤을때마다의 point feature

        return voxel_dict
