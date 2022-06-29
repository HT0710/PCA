from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data import read_data


def pca_data():
    dataset = read_data("Wage_data.csv").load_data()
    sc = StandardScaler()
    scaled_data = sc.fit_transform(dataset)
    pca = PCA(n_components=2)
    return pca.fit_transform(scaled_data)