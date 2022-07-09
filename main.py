from data import *

filename = "Wage_data.csv"
dataset = load_data(filename)
dataset = dataset.loc[:, ['nearc4', 'educ', 'age', 'black', 'wage', 'IQ', 'married', 'exper']]
# print(dataset[dataset.wage > 1000]['wage'].count()/1600)
# dataset = dataset.drop(dataset[dataset.wage > 1000].index)
# print(dataset)
X = dataset.loc[:, dataset.columns != 'wage'].values
y = dataset['wage'].values


LR = LinearRegression()


LR.fit(X, y)
# print(LR.coef_)

# dự đoán giá id 1 với wage = 721
print(dataset.loc[1:1], '\n')
pred = LR.predict([[0, 12, 34, 0, 103, 1, 16]])[0]
print("Before PCA")
print(f"Predicted wage: {round(pred)}")
print(f"Different: {round(abs(1 - pred / 721) * 100, 2)}%")


X = pca_data(X)
# print(X)

LR.fit(X, y)
# print(LR.coef_)

pred_pca = LR.predict([[2.79766258, -1.392667]])[0]
print("\nAfter PCA")
print(f"Predicted wage: {round(pred_pca)}")
print(f"Different: {round(abs(1 - pred_pca / 721) * 100, 2)}%")

## So sánh tỉ lệ trước và sau khi PCA
print(f"\nBefore vs After PCA: {round(abs(1 - pred_pca / pred) * 100, 2)}%")
