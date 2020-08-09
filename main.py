import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from torchfm.model.fm import FactorizationMachineModel
from BaseModel import FMModel
from FMRT import FMRT
from dataset import FeatureDataset
import os

np.random.seed(666)
torch.manual_seed(666)
torch.cuda.manual_seed(666)

def parse_args():
    parser = argparse.ArgumentParser(description="SeqFM.")
    parser.add_argument('--baseline', default='fm',
                        help='model name')
    parser.add_argument('--data', default='frappe',
                        help='[frappe]')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size.')
    parser.add_argument('--maxlen', type=int, default=20,
                        help='Max length of seqs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--reload', action='store_true',
                        help='restore saved params if true')
    parser.add_argument('--eval', action='store_true',
                        help='only eval once, non-train')
    parser.add_argument('--save', action='store_true',
                        help='if save model or not')
    parser.add_argument('--savepath',
                        help='for customization')
    parser.add_argument('--cuda', default='1',
                        help='gpu No.')
    parser.add_argument('--rt', action='store_true',
                        help='use FM-RT')
    parser.add_argument('--q', type=int, default=2,
                        help='Max num of perturbations')
    return parser.parse_args()


dataset_path = '../benchmarks/datasets/'
args = parse_args()
if not args.savepath:
    args.savepath = 'checkpoints/' + args.baseline + '_' + args.data
if not args.eval and args.rt:
    args.savepath += '_rt'
args.savepath += '.model'
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda


train = FeatureDataset(dataset_path + args.data + '/train.txt', maxlen=args.maxlen)
valid = FeatureDataset(dataset_path + args.data + '/valid.txt', maxlen=args.maxlen)
test = FeatureDataset(dataset_path + args.data + '/test.txt', maxlen=args.maxlen)

user_dim = max(train.n_user, valid.n_user, test.n_user)
item_dim = max(train.n_item, valid.n_item, test.n_item)
feature_dim = max(train.n_fea, valid.n_fea, test.n_fea)
feature_len = train.feature.shape[-1] - args.maxlen - 2
print(user_dim, item_dim, feature_dim, len(train) + len(valid) + len(test) * 2)

field_dims = [0] * (args.maxlen-1) + [item_dim] * 2 + [user_dim] + [0] * (feature_len - 1) + [feature_dim]
if args.rt:
    task = 'rt'
    small_better = [False, False]
    args.reload = True
    model = FMRT(FactorizationMachineModel(field_dims, 64),
                 seq_dim=item_dim, q=args.q, seq_len=args.maxlen, save_path=args.savepath, use_cuda=True)
else:
    task = 'classification'
    small_better = [False]
    model = FMModel(FactorizationMachineModel(field_dims, 64))

ref = -1

test_bs = min(args.batch_size, 64)
train = DataLoader(train, batch_size=args.batch_size, shuffle=True)
valid = DataLoader(valid, batch_size=test_bs)
test = DataLoader(test, batch_size=test_bs)

fn = {'classification': (model.fit_nll_neg, model.test_classify),
      'rt': (model.fit_nll_neg_rt, model.test_classify_acc)}

model.train_test(fn[task][0], fn[task][1], train, valid, test, only_eval=args.eval,
                 savepath=args.savepath if args.baseline == 'fm' else None, reload=args.reload,
                 data=args.data, n_epochs=args.epochs, lr=args.lr, topk=task=='rank',
                 small_better=small_better, ref=ref)
