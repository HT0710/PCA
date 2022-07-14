import random
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

file_name = "_Wage_data.csv"


def setup_data():
    data = DATA(file_name)
    data.pca()

    dataset = data.get_dataset()
    pca_data = data.get_pca_data()

    # nearc4,educ,age,weight,step14,black,wage,enroll,KWW,IQ,married,libcrd14,exper,lwage,expersq
    feature = ['nearc4', 'educ', 'age', 'black', 'wage', 'IQ', 'lwage']
    dataset = dataset.loc[:, feature]

    X = dataset.loc[:, dataset.columns != 'wage'].values
    y = dataset['wage'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    X_pca = pca_data
    X_pca_train, X_pca_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=0)

    return X_train, X_test, X_pca_train, X_pca_test, y_train, y_test, feature


def history():
    his = pd.read_csv('.history.csv')
    print(his)
    # print(his[his['Before'] <= 10])


class DATA:
    def __init__(self, data):
        self.dataset = self._load_data(data)
        self.pca_data = None

    def get_dataset(self):
        return self.dataset

    def get_pca_data(self):
        return self.pca_data

    @staticmethod
    def _load_data(data):
        # Load data
        data = pd.read_csv(data)
        # Drop row contain Blank or Null or NaN value
        data = data.dropna()
        # Reset index in dataset
        dataset = data.reset_index()

        return dataset[dataset.columns.difference(['index'])]

    def pca(self):
        sc = StandardScaler()
        scaled_data = sc.fit_transform(self.dataset)
        pca = PCA(n_components=2)
        pca = pca.fit_transform(scaled_data)

        self.pca_data = pca
