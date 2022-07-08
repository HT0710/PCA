import pandas as pd
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_data(filename):
    # Load data
    raw_data = pd.read_csv(filename)
    # Drop row contain Blank or Null or NaN value
    raw_data = raw_data.dropna()
    # Reset index in dataset
    dataset = raw_data.reset_index()

    return dataset[dataset.columns.difference(['index'])]


def pca_data(dataset):
    sc = StandardScaler()
    scaled_data = sc.fit_transform(dataset)
    pca = PCA(n_components=2)
    pca = pca.fit_transform(scaled_data)
    return pca