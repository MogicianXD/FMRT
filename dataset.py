import numpy as np
from torch.utils.data.dataset import Dataset

class FeatureDataset(Dataset):
    def __init__(self, datapath, neg_num=2, maxlen=0):
        self.feature = np.loadtxt(datapath, delimiter='\t', dtype=int)
        self.n_item = self.feature[:, :maxlen + 1].max() + 1
        self.n_user = self.feature[:, maxlen + 1].max() + 1
        self.n_fea = self.feature[:, maxlen + 2:].max() + 1
        self.feature = np.reshape(self.feature, (-1, 1 + neg_num, self.feature.shape[-1]))

    def __getitem__(self, indice):
        return self.feature[indice]

    def __len__(self):
        return len(self.feature)