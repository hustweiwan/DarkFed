import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    https://github.com/adambielski/siamese-triplet/blob/master/losses.py
    """

    def __init__(self, margin=1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class CompositeLoss(nn.Module):
    all_mode = ("cosine", "hinge", "contrastive")

    def __init__(self, rules, simi_factor, mode, size_average=True, *simi_args):
        """
        rules: a list of the attack rules, each element looks like (trigger1, trigger2, ..., triggerN, target)
        """
        super(CompositeLoss, self).__init__()
        self.rules = rules
        self.size_average = size_average
        self.simi_factor = simi_factor

        self.mode = mode
        if self.mode == "cosine":
            self.simi_loss_fn = nn.CosineEmbeddingLoss(*simi_args)
        elif self.mode == "hinge":
            self.pdist = nn.PairwiseDistance(p=1)
            self.simi_loss_fn = nn.HingeEmbeddingLoss(*simi_args)
        elif self.mode == "contrastive":
            self.simi_loss_fn = ContrastiveLoss(*simi_args)
        else:
            assert self.mode in all_mode

    def forward(self, y_hat, y):

        ce_loss = nn.CrossEntropyLoss()(y_hat, y)

        simi_loss = 0
        for rule in self.rules:
            mask = torch.BoolTensor(size=(len(y),)).fill_(0).cuda()
            for trigger in rule:
                mask |= y == trigger

            if mask.sum() == 0:
                continue

            # making an offset of one element
            y_hat_1 = y_hat[mask][:-1]
            y_hat_2 = y_hat[mask][1:]
            y_1 = y[mask][:-1]
            y_2 = y[mask][1:]

            if self.mode == "cosine":
                class_flags = (y_1 == y_2) * 1 + (y_1 != y_2) * (-1)
                loss = self.simi_loss_fn(y_hat_1, y_hat_2, class_flags.cuda())
            elif self.mode == "hinge":
                class_flags = (y_1 == y_2) * 1 + (y_1 != y_2) * (-1)
                loss = self.simi_loss_fn(self.pdist(y_hat_1, y_hat_2), class_flags.cuda())
            elif self.mode == "contrastive":
                class_flags = (y_1 == y_2) * 1 + (y_1 != y_2) * 0
                loss = self.simi_loss_fn(y_hat_1, y_hat_2, class_flags.cuda())
            else:
                assert self.mode in all_mode

            if self.size_average:
                loss /= y_hat_1.shape[0]

            simi_loss += loss

        return ce_loss + self.simi_factor * simi_loss