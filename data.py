import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_data(filename):
    # Load data
    raw_data = pd.read_csv(filename)
    # Drop row contain Blank or Null or NaN value
    raw_data = raw_data.dropna()
    # Reset index in dataset
    raw_data = raw_data.reset_index()
    # Select main Columns
    dataset = raw_data.loc[:,
              ['nearc4', 'educ', 'age', 'black', 'wage', 'IQ', 'married', 'exper', 'lwage', 'expersq']].copy()
    return dataset


def pca_data(filename):
    dataset = load_data(filename)
    sc = StandardScaler()
    scaled_data = sc.fit_transform(dataset)
    pca = PCA(n_components=2)
    pca = pca.fit_transform(scaled_data)
    return pca
