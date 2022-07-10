import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from bioinfokit.analys import get_data
from bioinfokit.visuz import cluster

raw_data = pd.read_csv("../_Wage_data.csv")
raw_data = raw_data.dropna()
raw_data = raw_data.reset_index()
# dataset = RAW_DATA.loc[:,
#           ['nearc4', 'educ', 'age', 'black', 'wage', 'IQ', 'married', 'exper', 'lwage', 'expersq']].copy()
dataset = raw_data
for i, j in zip([1000, 500, 5], range(3, 0, -1)):
    dataset['wage'] = np.where(dataset['wage'] > i, j, dataset['wage'])
# print(dataset)
feature = dataset.iloc[:, dataset.columns != 'wage']
target = dataset.iloc[:, 26].to_numpy()

X = feature
target = target

X_st = StandardScaler().fit_transform(X)
pca_out = PCA().fit(X_st)

# component loadings
loadings = pca_out.components_
# print(loadings)

# get biplot
pca_scores = PCA().fit_transform(X_st)
cluster.biplot(show=True, dim=(20,20), dotsize=10, cscore=pca_scores, loadings=loadings, labels=X.columns.values, var1=round(pca_out.explained_variance_ratio_[0]*100, 2),
    var2=round(pca_out.explained_variance_ratio_[1]*100, 2), colorlist=target)