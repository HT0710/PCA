from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from data import *

X, y = setup_data(feature)


def test(X, num_components):
    # Step-1
    X_meaned = X - np.mean(X, axis=0)

    # Step-2
    cov_mat = np.cov(X_meaned, rowvar=False)

    # Step-3
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

    # Step-4
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]

    # Step-5
    eigenvector_subset = sorted_eigenvectors[:, 0:num_components]

    # Step-6
    X_reduced = np.dot(eigenvector_subset.transpose(), X_meaned.transpose()).transpose()

    return X_reduced

def plot(data, size=(5, 5)):
    per_var = np.round(data * 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
    plt.figure(figsize=size)
    plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
    plt.ylabel('Percentage of explained variance')
    plt.xlabel('Principal component')
    plt.show()
    print(per_var)


def setup_pca(X):
    pca_data = pca(X)
    return pca_data.get_pca_dataset()


class pca:
    def __init__(self, dataset):
        self.dataset = dataset
        self.scaled_data = self._scaled_data()
        self.pca_dataset = self._fit_transform()
        self.variance_ratio = self._variance_ratio()

    def get_dataset(self):
        return self.dataset

    def get_scaled_data(self):
        return self.scaled_data

    def get_pca_dataset(self):
        return self.pca_dataset

    def get_variance_ratio(self):
        return self.variance_ratio

    def _scaled_data(self):
        sc = StandardScaler()
        self.scaled_data = sc.fit_transform(self.dataset)

        return self.scaled_data

    def _fit_transform(self):
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(self.scaled_data)
        self.pca_dataset = pca_data

        return self.pca_dataset

    def _variance_ratio(self):
        pca = PCA()
        pca.fit_transform(self.scaled_data)
        self.variance_ratio = pca.explained_variance_ratio_

        return self.variance_ratio
