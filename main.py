import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
raw_data = pd.read_csv("Wage_data.csv")
# Drop row contain Blank or Null or NaN value
raw_data = raw_data.dropna()
# Reset index in dataset
raw_data = raw_data.reset_index()
# Select main Columns
dataset = raw_data.loc[:,
          ['nearc4', 'educ', 'age', 'black', 'wage', 'IQ', 'married', 'exper', 'lwage', 'expersq']].copy()

for i, j in zip([2000, 1500, 1000, 500, 5], range(5, 0, -1)):
    dataset['wage'] = np.where(dataset['wage'] > i, j, dataset['wage'])

X = dataset.iloc[:, dataset.columns != 'wage'].values
y = dataset.iloc[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

pca = PCA(n_components=2)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

while True:
    check = int(input("Train(1) - Test(2): "))
    if check not in (1, 2):
        break

    (X_set, y_set) = (X_test, y_test) if check == 2 else (X_train, y_train)

    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_set, y_set)

    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1,
                                   stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1,
                                   stop=X_set[:, 1].max() + 1, step=0.01))

    plt.figure(figsize=(20, 20))

    # Basic contour plot
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),
                                                      X2.ravel()]).T).reshape(X1.shape), alpha=0.9, cmap=cm.gray)

    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], s=15, label=['> 0', '> 500', '> 1000', '> 1500', '> 2000'][i])

    # title for scatter plot
    plt.title('Principal Component Analysis')
    plt.xlabel('PC1')  # for Xlabel
    plt.ylabel('PC2')  # for Ylabel
    plt.legend()

    # show scatter plot
    plt.show()