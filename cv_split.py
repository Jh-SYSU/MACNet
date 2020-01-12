

'''
k fold cross validation split
'''
import os
import pandas as pd
from sklearn.model_selection import KFold

n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=666)

path = '/data/yuedongyang/raojh/ODIR/'
df = pd.read_excel(os.path.join(path,'ODIR-5K_training-Chinese.xlsx'))

i = 1
for train_idx, val_idx in kf.split(df):
    fold_train, fold_val = df.iloc[train_idx], df.iloc[val_idx]
    fold_train.to_csv(path+'train_%d.csv'%i, index=False)
    fold_val.to_csv(path+'valid_%d.csv'%i, index=False)
    print('[fold %d] num_train: %d num_valid: %d' % (i,
        fold_train.shape[0], fold_val.shape[0]))
    i += 1

print('Done.')

