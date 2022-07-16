import pandas as pd
from sklearn.model_selection import train_test_split

# nearc4,educ,age,weight,step14,black,wage,enroll,KWW,IQ,married,libcrd14,exper,lwage,expersq
feature = ['nearc4', 'educ', 'age', 'black', 'wage', 'IQ', 'lwage']
pred = 'wage'


def setup_data(file_name):
    data = DATA(file_name)
    dataset = data.load_data()

    dataset = dataset.loc[:, feature]

    X = dataset.loc[:, dataset.columns != pred].values
    y = dataset[pred].values

    return X, y


def train_test_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=0)


class DATA:
    def __init__(self, data):
        self.raw_data = data
        self.dataset = self.load_data()

    def get_dataset(self):
        return self.dataset

    def set_data(self, data):
        self.raw_data = data

    def load_data(self):
        data = pd.read_csv(self.raw_data)
        data = data.dropna()
        dataset = data.reset_index()
        self.dataset = dataset[dataset.columns.difference(['index'])]

        return self.dataset
