import numpy as np
from sklearn.metrics import f1_score
import torch
import scipy as sp

def get_score(y_true, y_pred):
    return 0
#     score = sp.stats.pearsonr(y_true, y_pred)[0]
#     return score


import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    """

    def __init__(self, eps=0.0):
        """
        Constructor.
        Args:
            eps (float, optional): Smoothing value. Defaults to 0.
        """
        super(SmoothCrossEntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets, w=None):
        """
        Computes the loss.
        Args:
            inputs (torch tensor [bs x n]): Predictions.
            targets (torch tensor [bs x n] or [bs]): Targets.
        Returns:
            torch tensor [bs]: Loss values.
        """
        mask = (targets == -1)
        targets = torch.clamp(targets, min=0)

        if len(targets.size()) == 1:  # to one hot
            targets = torch.zeros_like(inputs).scatter(1, targets.view(-1, 1).long(), 1)

        if self.eps > 0:
            n_class = inputs.size(1)
            targets = targets * (1 - self.eps) + (1 - targets) * self.eps / (
                n_class - 1
            )

        loss = -targets * F.log_softmax(inputs, dim=1)

        if len(loss.size()) == 2 and len(mask.size()) == 1:  # In case it was one-hot encoded
            mask = mask.unsqueeze(-1).repeat(1, loss.size(1))
        loss = loss.masked_fill(mask, 0)

        if w is None:
            loss = loss.sum(-1)
        else:
            if len(loss.size()) == 3:
                w = w.unsqueeze(1)
            elif len(loss.size()) == 4:
                w = w.unsqueeze(1).unsqueeze(1)
            elif len(loss.size()) == 5:
                w = w.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            loss = (loss * w).sum(-1)

        return loss


class SigmoidMSELoss(nn.Module):
    """
    Sigmoid on preds + MSE
    """
    def forward(self, inputs, targets):
        inputs = inputs.view(targets.size()).sigmoid()
        mask = (targets == -1)
        loss = ((inputs * 100 - targets * 100) ** 2)
        # loss = torch.abs(inputs * 100 - targets * 100)
        loss = loss.masked_fill(mask, 0)
        return loss.mean(-1)


class SeriesLoss(nn.Module):
    """
    Custom loss function for series predictions.
    """
    def __init__(self, eps=0.0, weighted=False):
        """
        Constructor.

        Args:
            eps (float, optional): Smoothing factor for cross-entropy loss. Defaults to 0.0.
            weighted (bool, optional): Flag to apply class-weighted loss. Defaults to False.
        """
        super().__init__()
        self.eps = eps
        self.ce = SmoothCrossEntropyLoss(eps=eps)
        self.weighted = weighted

    def forward(self, inputs, targets):
        if len(targets.size()) == 2:
            targets = targets.view(-1)  # bs * n_classes
        else:
            targets = targets.view(-1, 3)  # bs * n_classes x 3
        inputs = inputs.view(inputs.size(0), -1, 3).reshape(-1, 3)
        # bs x n_classes * 3 -> bs * n_classes x 3

        w = torch.pow(2, targets) if self.weighted else 1  # 1, 2, 4

        loss = self.ce(inputs, targets) * w
        return loss


class LogLoss(nn.Module):
    """
    Cross-entropy loss without softmax
    """
    def forward(self, inputs, targets):
        if len(targets.size()) == 1:  # to one hot
            targets = torch.zeros_like(inputs).scatter(1, targets.view(-1, 1).long(), 1)
        loss = -targets * torch.log(inputs)
        return loss.sum(-1)


class StudyLoss(nn.Module):
    """
    Custom loss function for patient predictions.

    Attributes:
        eps (float): Smoothing factor for cross-entropy loss.
        weighted (bool): Flag to apply class-weighted loss.
        use_any (bool): Flag to include 'any' label in the loss calculation.
        bce (nn.BCEWithLogitsLoss): BCE loss for bowel & extravasation.
        ce (SmoothCrossEntropyLoss): CE loss for spleen, liver & kidney.
    """
    def __init__(self, eps=0.0, weighted=True, use_any=True):
        """
        Constructor.

        Args:
            eps (float, optional): Smoothing factor for cross-entropy loss. Defaults to 0.0.
            weighted (bool, optional): Flag to apply class-weighted loss. Defaults to True.
            use_any (bool, optional): Include 'any' label in the loss calculation. Defaults to True.
        """
        super().__init__()
        self.eps = eps
        self.ce = SmoothCrossEntropyLoss(eps=eps)
        # self.ce  = LogLoss()
        self.weighted = weighted
        self.use_any = use_any

    def forward(self, inputs, targets):
        """
        Forward pass for the PatientLoss class.

        Args:
            inputs (torch.Tensor): Model predictions of shape (batch_size, num_classes).
            targets (torch.Tensor): Ground truth labels of shape (batch_size, num_classes).

        Returns:
            torch.Tensor: Loss value.
        """
        assert (targets.size(1) == 25) and (len(targets.size()) == 2), "Wrong target size"
        assert (inputs.size(1) == 25) and (len(inputs.size()) == 3), "Wrong input size"
        bs = targets.size(0)
        w = torch.pow(2, targets) if self.weighted else 1

        # print(inputs.view(-1, 3))

        loss = self.ce(inputs.reshape(-1, 3), targets.reshape(-1)).reshape(bs, -1)
        print(loss.mean(0))
        print(targets.float().mean(0))

        loss_scs, w_scs, y_scs = loss[:, :5].flatten(), w[:, :5].flatten(), targets[:, :5].flatten()
        loss_nfn, w_nfn, y_nfn = loss[:, 5:15].flatten(), w[:, 5:15].flatten(), targets[:, 5:15].flatten()
        loss_ss, w_ss, y_ss = loss[:, 15:].flatten(), w[:, 15:].flatten(), targets[:, 15:].flatten()

        missed_ratio_scs = (y_scs==-100).float().mean()
        missed_ratio_nfn = (y_nfn==-100).float().mean()
        missed_ratio_ss = (y_ss==-100).float().mean()
        num_studies = inputs.shape[0]

        loss_scs = (loss_scs * w_scs).sum() / w_scs.sum()
        loss_nfn = (loss_nfn * w_nfn).sum() / w_nfn.sum()
        loss_ss = (loss_ss * w_ss).sum() / w_ss.sum()

        if not self.use_any:
            return (loss_scs + loss_nfn + loss_ss) / 3

        any_target = targets[:, :5].amax(1)
        any_pred = inputs[:, :5].softmax(-1)[:, :, 2].amax(1)

        any_w = torch.pow(2, any_target) if self.weighted else 1
        any_target = (any_target == 2).long()

        # print(any_target, any_pred)

        any_loss = - any_target * torch.log(any_pred) - (1 - any_target) * torch.log(1 - any_pred)
        any_loss = torch.nan_to_num(any_loss,nan=0.0, posinf=0.0, neginf=0.0)
        any_loss = (any_w * any_loss).sum() / any_w.sum()

        loss_scs = torch.nan_to_num(loss_scs)
        loss_nfn = torch.nan_to_num(loss_nfn)
        loss_ss = torch.nan_to_num(loss_ss)
        any_loss = torch.nan_to_num(any_loss)

        return loss_scs, loss_nfn, loss_ss, any_loss, missed_ratio_scs, missed_ratio_nfn, missed_ratio_ss, num_studies


def calc_metric(cfg, pp_out, val_df, pre="val"):

    study_loss = self = StudyLoss()
    inputs, targets = pp_out['logits'],pp_out['target']

    loss_scs, loss_nfn, loss_ss, any_loss, \
        missed_ratio_scs, missed_ratio_nfn, missed_ratio_ss, num_studies \
                    = study_loss(pp_out['logits'],pp_out['target'])

    loss = (loss_scs + loss_nfn + loss_ss + any_loss) / 4
    scores = {'score_neural_foraminal_narrowing':loss_nfn,
             'score_spinal_canal_stenosis':loss_scs,
              'score_subarticular_stenosis':loss_ss,
              'score_spinal_any_severe':any_loss,
              'missed_ratio_scs' : missed_ratio_scs,
              'missed_ratio_nfn': missed_ratio_nfn,
              'missed_ratio_ss': missed_ratio_ss,
              #'missing_studies': len(val_df.study_id.unique()) - num_studies,
              'num_studies': float(num_studies)}

    if hasattr(cfg, "neptune_run"):
        for s in scores:
            cfg.neptune_run[f"{pre}/{s}/"].log(scores[s], step=cfg.curr_step)
            print(f"{pre} {s}: {scores[s]:.6}")
    return loss
