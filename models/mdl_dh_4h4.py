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
        self.backbone_out = backbone_out

        rnn_dict = dict(
                   batch_first=True,
                   num_layers=1,
                   dropout=0.0,
                   bidirectional=True)
        self.rnn = nn.LSTM(self.backbone_out, self.cfg.hidden_dim, **rnn_dict)

        if cfg.pool == "gem":
            self.global_pool = GeM(p_trainable=cfg.gem_p_trainable)
        elif cfg.pool == "identity":
            self.global_pool = torch.nn.Identity()
        elif cfg.pool == "avg":
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.head = torch.nn.Linear(self.cfg.hidden_dim * 2, self.n_classes)
        self.loss_fn = torch.nn.L1Loss(reduction = 'none')
        print(f'Net parameters: {human_format(count_parameters(self))}')

    def forward(self, batch):

        x = batch['input']
        mask = batch['mask']

        x = x[mask==1]

        x = self.normalization(x)
        x = self.backbone(x)
        x = self.global_pool(x)
        x = x[:,:,0,0]

        #x_emb = torch.zeros_like(x[:,:self.level_emb.embedding_dim])

        b, slen = batch['input'].shape[:2]
        x_hidden = torch.zeros_like(batch['input'].view(b*slen, -1)[:,:x.shape[-1]]).to(x.dtype)
        # x_hidden =  torch.zeros((*batch['input'].shape[:2], x.shape[-1])).to(x.dtype).to(x.device)
        x_hidden[mask.view(-1)==1] = x
        x_hidden = x_hidden.view(b,slen, -1)

        x_h, _ = self.rnn(x_hidden)
        logits = self.head(x_h)
        outputs = {}

        if 'target' in batch:

            y = batch['target']
            y_mask = batch['target_mask']
            y, y_mask, logits2 = y[:,:,[0,1,2,2]].clone(), \
                    y_mask[:,:,[0,1,2,2]].clone(), logits[:,:,[0,1,2,2]].clone()

            wt = (y_mask.max(2).values.sum(1)  / 20).unsqueeze(1).repeat(1, logits.shape[1] )


            loss_xy = self.loss_fn(logits2,y)
            outputs['loss_l1'] = loss_xy[y_mask==1].mean()

            loss_xy_wt = loss_xy * wt[:,:,None]
            outputs['loss'] = loss_xy_wt[y_mask==1].mean()


            if y_mask[:,:,2:].sum()==0:
                y[:,:,2:], y_mask[:,:,2:], logits2[:,:,2:] = y[:,:,:2], y_mask[:,:,:2], logits2[:,:,:2]
            elif y_mask[:,:,:2].sum()==0:
                y[:,:,:2], y_mask[:,:,:2], logits2[:,:,:2] = y[:,:,2:], y_mask[:,:,2:], logits2[:,:,2:]


            loss_xy2 = self.loss_fn(logits2,y)
            outputs['loss_l1_nfn'] = loss_xy2[:,:,:2][y_mask[:,:,:2]==1].mean()
            outputs['loss_l1_scs'] = loss_xy2[:,:,2:][y_mask[:,:,2:]==1].mean()

            # loss_xy = loss_xy ** 2 # loss mse

            #outputs['loss_nfn'] = loss_xy[:,:,:2][y_mask[:,:,:2]==1].mean() ** 0.5
            #outputs['loss_scs'] = loss_xy[:,:,2:][y_mask[:,:,2:]==1].mean() ** 0.5


        if not self.training:
            series_ids = batch['series_id'].unsqueeze(1).repeat(1, logits.shape[1] )
            outputs['series_id'] = series_ids[mask==1]
            outputs['instance_number'] = batch['instance_numbers'][mask==1]
            outputs["logits"] = logits[mask==1]

        return outputs