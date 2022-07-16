import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from data import setup_data

# Show
PC_ratio = True


def main():
    X, y = setup_data('csv/_Wage_data.csv')
    pca_data = _PCA(X)

    # pca = PCA()
    # pca.fit_transform(pca_data.get_scaled_data())
    # print(pca.)

    if PC_ratio:
        per_var = np.round(pca_data.variance_ratio * 100, decimals=1)
        print(per_var)
        labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
        plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
        plt.ylabel('Percentage %')
        plt.xlabel('Principal component')
        plt.title('PC ratio')
        plt.show()


def train_test_pca(X, y):
    pca_data = _PCA(X)
    X_pca = pca_data.get_pca_dataset()
    return train_test_split(X_pca, y, test_size=0.2, random_state=0)


class _PCA:
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


if __name__ == '__main__':
    main()
