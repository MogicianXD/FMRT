import numpy as np
import pandas as pd

df = pd.read_csv('frappe.csv', delimiter='\t')
del df['cnt']
df['item'] += 1
# df = df[['item', 'user'] + list(df.columns[2:])]

static_dim = 0
for col in df.columns[2:]:
    unique = df[col].unique()
    id = pd.Series(index=unique, data=np.arange(static_dim, static_dim + len(unique)))
    df[col] = df[col].apply(lambda x: id[x])
    static_dim += len(id)
    print(static_dim)

def del_infreq(key, threshold):
    cnts = df[key].value_counts()
    left = cnts[cnts >= threshold]
    return df[df[key].isin(left.index)]

# df = del_infreq('item', 10)
# df = del_infreq('user', 10)

indice = df.index.to_list()
np.random.shuffle(indice)
data = {}
data['train'] = df.loc[indice[: int(0.8 * len(indice))]]
data['valid'] = df.loc[indice[int(0.8 * len(indice)): int(0.9 * len(indice))]]
data['test'] = df.loc[indice[int(0.9 * len(indice)):]]

neg_num = 2
all_items = set(range(1, df['item'].max()))
for type in ['train', 'valid', 'test']:
    with open(type + '.txt', 'w') as f:
        for uid, item in data[type].groupby('user'):
            pre = []
            history = set(item['item'])
            candidates = list(all_items - history)
            for _, record in item.iterrows():
                seq = pre + [0] * (20 - len(pre))
                out_line = '\t'.join([str(i) for i in seq]) + '\t{}' + '\t' + str(uid) + '\t' \
                           + '\t'.join([str(v) for v in record.values[2:]]) + '\n'
                f.write(out_line.format(record['item']))
                pre.append(record['item'])
                pre = pre[-20:]
                for neg_i in range(neg_num):
                    f.write(out_line.format(np.random.choice(candidates)))



