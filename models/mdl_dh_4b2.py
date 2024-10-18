import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torch.distributions import Beta
import timm
import numpy as np

class ImageNormalization(nn.Module):

    def __init__(self, mode):
        super(ImageNormalization, self).__init__()

        self.mode = mode

    def forward(self, x):
        with torch.no_grad():
            if self.mode == 'image':

                x = (x - x.mean((1,2,3),keepdim=True)) / (x.std((1,2,3),keepdim=True) + 1e-4)
                x = x.clamp(-20,20)

            elif self.mode == 'simple':
                x = x / 255.

            elif self.mode == "channel":

                x = (x - x.mean((2,3),keepdim=True)) / (x.std((2,3),keepdim=True) + 1e-4)
                x = x.clamp(-20,20)

            elif self.cfg.normalization == "min_max":

                x = x - x.amin((1,2,3),keepdims=True)
                x = x / (x.amax((1,2,3),keepdims=True) + 1e-4)

            elif self.cfg.normalization == 'imagenet':

                x = x / 255.
                x = x - torch.tensor((0.485, 0.456, 0.406))[None,:,None,None]
                x = x / torch.tensor((0.229, 0.224, 0.225))[None,:,None,None]
            else:
                raise Exception

        return x

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)
        return ret

    def __repr__(self):
        return (self.__class__.__name__  + f"(p={self.p.data.tolist()[0]:.4f},eps={self.eps})")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

'''
self = Net(cfg)
self(batch)
'''


class Net(nn.Module):

    def __init__(self, cfg):
        super(Net, self).__init__()

        self.cfg = cfg
        self.n_classes = self.cfg.n_classes
        self.normalization = ImageNormalization(cfg.image_normalization)
        self.backbone = timm.create_model(cfg.backbone,
                                          pretrained=cfg.pretrained,
                                          num_classes=0,
                                          global_pool="",
                                          in_chans=self.cfg.in_channels)


        if 'efficientnet' in cfg.backbone:
            backbone_out = self.backbone.num_features
        else:
            backbone_out = self.backbone.feature_info[-1]['num_chs']


        if cfg.pool == "gem":
            self.global_pool = GeM(p_trainable=cfg.gem_p_trainable)
        elif cfg.pool == "identity":
            self.global_pool = torch.nn.Identity()
        elif cfg.pool == "avg":
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.head = torch.nn.Linear(backbone_out, self.n_classes)
        self.head_mask = torch.nn.Linear(backbone_out, 1)
        self.loss_fn = torch.nn.L1Loss(reduction = 'none')
        self.loss_fn_mask = torch.nn.CrossEntropyLoss(reduction='none' )
        print(f'Net parameters: {human_format(count_parameters(self))}')

    def forward(self, batch):

        x = batch['input']#.unsqueeze(1)
        x = self.normalization(x)
        x = self.backbone(x)

        x = self.global_pool(x)
        x = x[:,:,0,0]

        logits = self.head(x)
        logits_mask = self.head_mask(x)
        outputs = {}

        if 'target' in batch:
            y = batch['target']
            y_mask = batch['target_mask']
            x_mask = batch['input_mask']
            loss_xy = self.loss_fn(logits,y)
            loss_mask = self.loss_fn(logits_mask.flatten(), x_mask).mean()

            loss_xy = \
                (loss_xy[:,range(0, loss_xy.shape[1], 2)] ** 2 + \
                    loss_xy[:,range(1, loss_xy.shape[1], 2)] ** 2) # ** 0.5

            target_mask = batch['target_mask'][:, range(0, batch['target_mask'].shape[1], 2)]

            loss_xy = loss_xy[target_mask==1].mean()
            loss = (loss_xy) * (1 - self.cfg.mask_loss) + (loss_mask) * (self.cfg.mask_loss)
            #outputs[f'loss_subarticular_stenosis'] = loss
            outputs['loss'] = loss
            outputs['loss_xy'] = loss_xy
            outputs['loss_mask'] = loss_mask
        outputs['series_id'] = batch['series_id']
        outputs['instance_number'] = batch['instance_numbers']
        if not self.training:
            outputs["logits"] = logits
            outputs["logits_mask"] = logits_mask

        return outputs