import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# def main():
#


def train_test_pca(X, y):
    pca_data = _PCA(X)
    X_pca = pca_data.get_pca_dataset()
    return train_test_split(X_pca, y, test_size=0.2, random_state=0)


class _PCA:
    def __init__(self, dataset):
        self.dataset = dataset
        self.pca_dataset = self._fit_transform()

    def get_dataset(self):
        return self.dataset

    def get_pca_dataset(self):
        return self.pca_dataset

    def _fit_transform(self):
        sc = StandardScaler()
        scaled_data = sc.fit_transform(self.dataset)
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)
        self.pca_dataset = pca_data

        return self.pca_dataset

    def pc_ratio(self):
        sc = StandardScaler()
        scaled_data = sc.fit_transform(self.dataset)
        pca = PCA()
        pca.fit_transform(scaled_data)
        per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
        labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
        plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
        plt.ylabel('Percentage of E V')
        plt.xlabel('Principal component')
        plt.title('Scree Plot')
        plt.show()

# if __name__ == '__pca__':
#     main()
