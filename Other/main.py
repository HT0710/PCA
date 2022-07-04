import numpy as np
from data import load_data
from matplotlib import pyplot as plt, cm
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = load_data("../Wage_data.csv")

# Classify target
for i, j in zip([1000, 500, 5], range(3, 0, -1)):
    dataset['wage'] = np.where(dataset['wage'] > i, j, dataset['wage'])

# Separate feature and target
feature = dataset.iloc[:, dataset.columns != 'wage'].values
target = dataset.iloc[:, 4].values


# PCA function
def pca(X, y):
    # Flip train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Standardized data
    sc = StandardScaler()
    scaled_X_Train = sc.fit_transform(X_train)
    Scaled_X_Test = sc.fit_transform(X_test)

    # Applied PCA to data
    pca = PCA(n_components=2)
    PCA_X_Train = pca.fit_transform(scaled_X_Train)
    PCA_X_Test = pca.fit_transform(Scaled_X_Test)

    return PCA_X_Train, PCA_X_Test, y_train, y_test


# Iris Data
# data = datasets.load_iris()
# X, y = data.data, data.target
# sc = StandardScaler()
# X_train = sc.fit_transform(X)
# PCA = PCA()
# X_train = PCA.fit_transform(X)
# y_train = y

# PCA variance ratio
# per_var = np.round(PCA.explained_variance_ratio_*100, decimals=1)
# labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
# plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
# plt.ylabel('Percentage of E V')
# plt.xlabel('Principal component')
# plt.title('Scree Plot')
# plt.show()

# main loop
while True:
    check = int(input("Train(1) - Test(2): "))
    if check not in (1, 2):
        break

    # Flip to train and test and PCA data
    (X_train, X_test, y_train, y_test) = pca(feature, target)
    # Choose train or test set
    (X_set, y_set) = (X_train, y_train) if check == 1 else (X_test, y_test)
    # X_set = X_train
    # y_set = y_train
    print(X_train.shape)

    # Classified to contour data
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_set, y_set)

    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1,
                                   stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1,
                                   stop=X_set[:, 1].max() + 1, step=0.01))

    # Create figure
    plt.figure(figsize=(20, 20))

    # Basic contour plot
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),
                                                      X2.ravel()]).T).reshape(X1.shape), alpha=0.9, cmap=cm.gray)

    # Scattered data
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], s=15, label=['> 0', '> 500', '> 1000'][i])

    # Title for scatter
    plt.title('Principal Component Analysis')
    plt.xlabel('PC1')  # for Xlabel
    plt.ylabel('PC2')  # for Ylabel
    plt.legend()

    # Show scatter plot
    plt.show()
