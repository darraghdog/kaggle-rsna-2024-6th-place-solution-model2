import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torch.distributions import Beta
import timm
from torch import Tensor
import numpy as np
from torch.nn.utils.rnn import pad_sequence



def create_sinusoidal_embeddings(n_pos, dim, pow_ = 10000):
    out = torch.zeros(n_pos, dim)
    position_enc = np.array([[pos / np.power(pow_, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out2 = torch.nn.Embedding(n_pos, dim)
    out2.load_state_dict({'weight':out})
    out2.requires_grad = False
    return out2


class Swish(nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()

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

def create_sinusoidal_embeddings(n_pos, dim, pow_ = 10000):
    out = torch.zeros(n_pos, dim)
    position_enc = np.array([[pos / np.power(pow_, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out2 = torch.nn.Embedding(n_pos, dim)
    out2.load_state_dict({'weight':out})
    out2.requires_grad = False
    return out2

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])



class FeedForwardModule(nn.Module):
    """
    Feed Forward Module follow pre-norm residual units and apply layer normalization within the residual unit
    and on the input before the first linear layer. This module also apply Swish activation and dropout, which helps
    regularizing the network.

    Args:
        encoder_dim (int): Dimension of squeezeformer encoder
        expansion_factor (int): Expansion factor of feed forward module.
        dropout_p (float): Ratio of dropout
    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor contains input sequences
    Outputs: outputs
        - **outputs** (batch, time, dim): Tensor produces by feed forward module.
    """

    def __init__(
        self,
        input_dim: int = 512,
        encoder_dim: int = 512,
        expansion_factor: int = 4,
        dropout_p: float = 0.1,
    ) -> None:
        super(FeedForwardModule, self).__init__()

        self.ffn1 = nn.Linear(input_dim, encoder_dim * expansion_factor, bias=True)
        self.act = Swish()
        self.do1 = nn.Dropout(p=dropout_p)
        self.ffn2 = nn.Linear(encoder_dim * expansion_factor, encoder_dim, bias=True)
        self.do2 = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.act(x)
        x = self.do1(x)
        x = self.ffn2(x)
        x = self.do2(x)

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self,
            embed_dim,
            num_head,
            dropout,
            batch_first,
        ):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim,
            num_heads=num_head,
            bias=True,
            add_bias_kv=False,
            kdim=None,
            vdim=None,
            dropout=dropout,
            batch_first=batch_first,
        )

    def forward(self, x, attn_mask = None, key_padding_mask = None):
        out, _ = self.mha(x,x,x, attn_mask = attn_mask, key_padding_mask=key_padding_mask)
        return out

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
        #for nm,p in self.backbone.named_parameters():
        #    p.requires_grad = False

        if 'efficientnet' in cfg.backbone:
            backbone_out = self.backbone.num_features
        else:
            backbone_out = self.backbone.feature_info[-1]['num_chs']
        self.backbone_out = backbone_out

        self.level_emb = torch.nn.Embedding(5, 32)

        self.backbone_out += self.level_emb.embedding_dim

        rnn_dict = dict(
                   batch_first=True,
                   num_layers=1,
                   dropout=0.0,
                   bidirectional=True)
        self.rnn_r_sag_t1 = nn.LSTM(self.backbone_out, self.cfg.hidden_dim, **rnn_dict)
        self.rnn_l_sag_t1 = nn.LSTM(self.backbone_out, self.cfg.hidden_dim, **rnn_dict)
        self.rnn_sag_t2 = nn.LSTM(self.backbone_out, self.cfg.hidden_dim, **rnn_dict)
        self.rnn_r_ss = nn.LSTM(self.backbone_out, self.cfg.hidden_dim, **rnn_dict)
        self.rnn_l_ss = nn.LSTM(self.backbone_out, self.cfg.hidden_dim, **rnn_dict)

        if cfg.pool == "gem":
            self.global_pool = GeM(p_trainable=cfg.gem_p_trainable)
        elif cfg.pool == "identity":
            self.global_pool = torch.nn.Identity()
        elif cfg.pool == "avg":
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.head_r_sag_t1 = torch.nn.Linear(self.cfg.hidden_dim * 2, len(cfg.classes))
        self.head_l_sag_t1 = torch.nn.Linear(self.cfg.hidden_dim * 2, len(cfg.classes))
        self.head_sag_t2 = torch.nn.Linear(self.cfg.hidden_dim * 2, len(cfg.classes))
        self.head_r_ss = torch.nn.Linear(self.cfg.hidden_dim * 2, len(cfg.classes))
        self.head_l_ss = torch.nn.Linear(self.cfg.hidden_dim * 2, len(cfg.classes))

        self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 4.0]))
        print(f'Net parameters: {human_format(count_parameters(self))}')
        # Flip side always --  Right -> Left

    def forward(self, batch):

        x = batch['input']
        y = batch['target'].clone()
        y_mask = y!=-100
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

        x_emb = torch.zeros_like(x_hidden[:,:,:self.level_emb.embedding_dim])
        x_emb[:] = self.level_emb (batch['level_num']).unsqueeze(1)

        x_hidden = torch.cat((x_hidden, x_emb), -1)
        condition_idx = batch['condition']

        losses = []
        logits = []
        series_ids = []
        study_ids = []
        conditions = []
        y_ls = []
        cond_ls = []

        outputs = {}

        if 'ss' in self.cfg.model_parts:
            x_h_r_ss, _ = self.rnn_r_ss(x_hidden)
            x_h_l_ss, _ = self.rnn_l_ss(x_hidden)
            x_h_r_ss  = torch.stack([i[m].mean(0) for i,m in zip(x_h_r_ss, mask.bool())])
            x_h_l_ss  = torch.stack([i[m].mean(0) for i,m in zip(x_h_l_ss, mask.bool())])
            logits_r_ss = self.head_r_ss(x_h_r_ss)
            logits_l_ss = self.head_l_ss(x_h_l_ss)
            loss_r_ss = self.loss_fn(logits_r_ss,y[:,1].long())
            loss_l_ss = self.loss_fn(logits_l_ss,y[:,0].long())
            losses.append( (loss_r_ss + loss_l_ss)/2 )
            logits_ss = torch.stack((logits_l_ss, logits_r_ss)).permute(1,0,2)
            conditions.append(2)
            logits .append( logits_ss.clone() )
            outputs['loss_l_ss'] = loss_l_ss
            outputs['loss_r_ss'] = loss_r_ss
            series_ids.append(batch['series_id'])
            study_ids.append(batch['study_id'])
            y_ls.append(y)
            cond_ls.append(batch['condition'])


        if 'nfn' in self.cfg.model_parts:
            x_h_r, _ = self.rnn_r_sag_t1(x_hidden)
            x_h_l, _ = self.rnn_l_sag_t1(x_hidden)
            x_h_r  = torch.stack([i[m].mean(0) for i,m in zip(x_h_r, mask.bool())])
            x_h_l  = torch.stack([i[m].mean(0) for i,m in zip(x_h_l, mask.bool())])
            logits_r = self.head_r_sag_t1(x_h_r)
            logits_l = self.head_l_sag_t1(x_h_l)

            y_nfn = batch['nfn_label']
            loss_r = self.loss_fn(logits_r,y_nfn[:,1].long())
            loss_l = self.loss_fn(logits_l,y_nfn[:,0].long())
            losses.append( (loss_r + loss_l)/2 )
            logits_t1 = torch.stack((logits_l,logits_r)).permute(1,0,2)
            outputs['loss_t1_l'] = loss_l
            outputs['loss_t1_r'] = loss_r
            conditions.append(0)
            logits .append( logits_t1.clone() )
            series_ids.append(batch['nfn_series_id'])
            study_ids.append(batch['study_id'])
            y_ls.append(y_nfn)
            cond_ls.append(batch['condition'] - 2)

        if 'scs' in self.cfg.model_parts:
            x_h_t2, _ = self.rnn_sag_t2(x_hidden)
            x_h_t2 = torch.stack([i[m].mean(0) for i,m in zip(x_h_t2, mask.bool())])
            logits_t2 = self.head_sag_t2(x_h_t2)

            y_scs = batch['scs_label']
            loss_t2 = self.loss_fn(logits_t2,y_scs[:,0].long())
            logits_t2 = torch.stack((logits_t2, logits_t2)).permute(1,0,2)
            outputs['loss_t2'] = loss_t2
            losses.append( loss_t2 )
            conditions.append(1)
            logits .append( logits_t2.clone() )
            series_ids.append(batch['scs_series_id'])
            study_ids.append(batch['study_id'])
            y_ls.append(y_scs.repeat(1,2))
            cond_ls.append(batch['condition'] - 1)

        series_ids = torch.cat(series_ids)
        study_ids = torch.cat(study_ids)
        y_out = torch.cat(y_ls)
        conditions = torch.cat(cond_ls)
        logits_all = torch.cat(logits)
        level_nums = batch['level_num'].repeat(len(cond_ls))

        filtidx = series_ids!=self.cfg.dummy_series_id

        #logits_all = torch.zeros_like(torch.cat(logits))
        #for cond, lgt in zip(conditions, logits):
        #    logits_all[condition_idx==cond] = lgt.clone()

        outputs['loss'] = torch.stack(losses).mean()
        outputs['series_id'] =  series_ids[ filtidx]
        outputs['study_id'] = study_ids[ filtidx]
        outputs['condition'] = conditions[ filtidx]
        outputs['level'] = level_nums[ filtidx]
        outputs['target'] = y_out[ filtidx]

        if not self.training:
            outputs["logits"] = logits_all[[ filtidx]]

        return outputs
