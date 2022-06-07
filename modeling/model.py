import torch
from .baseline import Baseline
from .nformer import NFormer
import torch.nn as nn
class nformer_model(nn.Module):
    def __init__(self, cfg, num_classes):
        super(nformer_model, self).__init__()
        self.backbone = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE)
        self.nformer = NFormer(cfg, num_classes)

    def forward(self,x,stage = 'encoder'):
        if stage == 'encoder':
            if self.training:
                score, feat = self.backbone(x)
                return score, feat
            else:
                feat = self.backbone(x)
                return feat

        elif stage == 'nformer':
            feat = self.nformer(x)
            return feat
