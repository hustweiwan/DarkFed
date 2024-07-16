import torch
from metrics.metric import Metric


class AccuracyMetric(Metric):

    def __init__(self, top_k=(1,)):
        self.top_k = top_k
        self.main_metric_name = 'Top-1'
        super().__init__(name='Accuracy', train=False)

    def compute_metric(self, outputs: torch.Tensor,
                       labels: torch.Tensor):
        """Computes the precision@k for the specified values of k"""
        #, specific = False
        max_k = max(self.top_k)
        batch_size = labels.shape[0]

        _, pred = outputs.topk(max_k, 1, True, True)
        pred = pred.t()
        #print(pred)
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        res = dict()
        for k in self.top_k:
            correct_k = correct[:k].view(-1).float().sum(0)
            res[f'Top-{k}'] = (correct_k.mul_(100.0 / batch_size)).item()
        # if specific:
        #     # 如果 specific 为 True，则只计算原类别为1的准确率
        #     mask = labels == 1
        #     for k in self.top_k:
        #         correct_k = correct[:k][mask].view(-1).float().sum(0)
        #         res[f'Top-{k}'] = (correct_k.mul_(100.0 / batch_size)).item()
        # else:
        # for k in self.top_k:
        #     correct_k = correct[:k].view(-1).float().sum(0)
        #     res[f'Top-{k}'] = (correct_k.mul_(100.0 / batch_size)).item()
        return res
