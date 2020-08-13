import time

import torch
from torch import optim
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from BaseModel import FMModel

class FMRT(FMModel):
    def __init__(self, FM, seq_dim, seq_len, save_path, q=2, use_cuda=True):
        super(FMRT, self).__init__(FM, use_cuda)
        self.seq_dim = seq_dim
        self.seq_len = seq_len
        self.q = q
        self.k = q * (q - 1) // 2
        self.save_path = save_path + '.q'
        candidate_idx = []
        for i in range(0, self.seq_dim-1):
            for j in range(i + 1, self.seq_dim-1):
                candidate_idx.append([i+1, j+1])
        self.candidate_idx = torch.tensor(candidate_idx, device=self.device, dtype=torch.int64)

    def fit_nll_neg_rt(self, input_batch, epsilon=1e-9):
        b = self._bound(input_batch[:, :, :self.seq_len])
        preds = torch.sigmoid(self.forward(input_batch) + b)
        cost = - torch.log(preds[:, 0] + epsilon).sum() - torch.log(1 - preds[:, 1:] + epsilon).sum()
        return cost / preds.shape[0]

    def _bound_info(self, maxq):
        vj = self.FM.embedding.embedding.weight[1: self.seq_dim] # n, k
        vi = self.FM.embedding.embedding.weight # d, k
        second_order = torch.matmul(vi, vj.transpose(-2, -1)) # d, n
        p = self.FM.linear.fc.weight[1: self.seq_dim].squeeze(-1) # n
        diag = torch.diag(second_order)
        diag_dsc, diag_dsc_idx = diag.topk(maxq, largest=True)
        diag_asc, diag_asc_idx = diag.topk(maxq, largest=False)
        diag_dsc_idx += 1
        diag_asc_idx += 1
        candidate = second_order[1: self.seq_dim]
        candidate = candidate[torch.ones(candidate.shape, device=self.device).tril(-1).bool()]
        maxq = (maxq - 1) * maxq // 2
        second_asc, second_asc_idx = candidate.topk(maxq, largest=False)
        second_dsc, second_dsc_idx = candidate.topk(maxq, largest=True)
        return p, diag_asc_idx, diag_asc, diag_dsc_idx, diag_dsc, \
               self.candidate_idx[second_asc_idx], second_asc, self.candidate_idx[second_dsc_idx], \
               second_dsc, second_order

    def _bound_p(self, input_batch, p, second_order, maxq):
        p = p + F.embedding(input_batch, second_order, padding_idx=0).sum(-2) # bs, 1+neg, n
        # b = torch.cat([p[:, 0].topk(self.q)[0].sum(-1),
        #                p[:, 1:].topk(self.q, largest=False)[0].sum(-1)], -1) # bs, 1+neg
        p_asc, p_asc_idx = p[:, 0].topk(maxq, largest=False)
        p_dsc, p_dsc_idx = p[:, 1:].topk(maxq, largest=True)
        p_asc_idx += 1
        p_dsc_idx += 1
        return p_asc, p_asc_idx, p_dsc, p_dsc_idx

    def _bound(self, input_batch):
        input_batch = input_batch.to(self.device)
        p, diag_asc_idx, diag_asc, diag_dsc_idx, diag_dsc, \
            second_asc_idx, second_asc, second_dsc_idx, second_dsc, second_order = self._bound_info(self.q+self.seq_len)
        p_asc, p_asc_idx, p_dsc, p_dsc_idx = self._bound_p(input_batch, p, second_order, maxq=self.q+self.seq_len)
        b = torch.zeros((input_batch.shape[0], input_batch.shape[1]), requires_grad=False, device=self.device)
        for bid in range(input_batch.shape[0]):
            history = input_batch[bid, 0: 1]
            b[bid, 0] += (self._bound_one(p_asc_idx[bid], p_asc[bid], history) +
                          self._bound_one(diag_asc_idx, diag_asc, history) +
                          self._bound_two(second_asc_idx, second_asc, history)).item()
            for i in range(1, input_batch.shape[1]):
                history = input_batch[bid, i: (i+1)]
                b[bid, i] += (self._bound_one(p_dsc_idx[bid, i-1], p_dsc[bid, i-1], history) +
                              self._bound_one(diag_dsc_idx, diag_dsc, history) +
                              self._bound_two(second_dsc_idx, second_dsc, history)).item()
        return b

    def _bound_one(self, idx, candidate, history):
        candidate = candidate[(idx.unsqueeze(1) == history).sum(1) == 0]
        return candidate[:self.q].sum()

    def _bound_two(self, idx, candidate, history):
        candidate = candidate[((idx[:, 1:] == history).sum(1) == 0) &
                              ((idx[:, :1] == history).sum(1) == 0)]
        return candidate[:self.k].sum()

    def _get_q_batch(self, input_batch, preds, p, diag_asc_idx, diag_asc, diag_dsc_idx, diag_dsc,
                     second_asc_idx, second_asc, second_dsc_idx, second_dsc, second_order):
        input_batch = input_batch.to(self.device)
        p_asc, p_asc_idx, p_dsc, p_dsc_idx = self._bound_p(input_batch, p, second_order, maxq=50)
        qs = []
        for bid in range(input_batch.shape[0]):
            history = input_batch[bid, 0]
            y = preds[bid, 0]
            if y > 0:
                qs.append(self._get_q(zip(p_asc_idx[bid], p_asc[bid]),
                                      zip(diag_asc_idx, diag_asc),
                                      zip(second_asc_idx, second_asc), history, y))
            for i in range(1, input_batch.shape[1]):
                history = input_batch[bid, i]
                y = preds[bid, i]
                if y < 0:
                    qs.append(self._get_q(zip(p_dsc_idx[bid, i-1], p_dsc[bid, i-1]),
                                          zip(diag_dsc_idx, diag_dsc),
                                          zip(second_dsc_idx, second_dsc), history, y))
        return qs

    def _get_q(self, p_iter, diag_iter, second_iter, history, y):
        q, b = 1, 0
        while q < self.seq_dim:
            b += self._pop(p_iter, history) + self._pop(diag_iter, history)
            for i in range(q - 1):
                b += self._pop_second(second_iter, history)
            if y.sign() * (y + b).sign() < 0:
                break
            q += 1
        return q - 1

    def _pop(self, iterator, history):
        while True:
            idx, v = next(iterator)
            if idx in history:
                continue
            else:
                break
        return v

    def _pop_second(self, iterator, history):
        while True:
            idx, v = next(iterator)
            if idx[0] in history or idx[1] in history:
                continue
            else:
                break
        return v

    def test_classify_acc(self, data):
        self.eval()
        ACC = 0
        qs = []
        with torch.no_grad():
            p, diag_asc_idx, diag_asc, diag_dsc_idx, diag_dsc, \
                second_asc_idx, second_asc, second_dsc_idx, second_dsc, second_order = self._bound_info(50)
            for input_batch in tqdm(data):
                pred = self.forward(input_batch)
                gt = torch.tensor([1.] + [-1.] * (input_batch.shape[-2] - 1), device=self.device)
                ACC += (pred * gt > 0).sum() / 3.
                qs += self._get_q_batch(input_batch[:, :, :self.seq_len], pred, p, diag_asc_idx, diag_asc, diag_dsc_idx, diag_dsc,
                                        second_asc_idx, second_asc, second_dsc_idx, second_dsc, second_order)
        avg_q = sum(qs) / len(qs)
        np.savetxt(self.save_path, np.array(qs, dtype=np.int32), '%d')
        ACC = ACC.item() / len(data.dataset)
        return ACC, avg_q

