from data import *

filename = "Wage_data.csv"
dataset = load_data(filename)
dataset = dataset.loc[:, ['nearc4', 'educ', 'age', 'black', 'wage', 'IQ', 'married', 'exper']]
# print(dataset[dataset.wage > 1000]['wage'].count()/1600)
# dataset = dataset.drop(dataset[dataset.wage > 1000].index)
# print(dataset)
X = dataset.loc[:, dataset.columns != 'wage'].values
y = dataset['wage'].values


regr = linear_model.LinearRegression()


regr.fit(X, y)
# print(regr.coef_)

# dự đoán giá id 1 với wage = 721
print(dataset.loc[1:1], '\n')
predict_values = regr.predict([[0, 12, 34, 0, 103, 1, 16]])[0]
A = predict_values
print("Before PCA")
print(f"Predicted wage: {round(predict_values)}")
print(f"Different: {round(abs(1-predict_values/721)*100, 2)}%")


X = pca_data(X)
# print(X)
regr.fit(X, y)
# print(regr.coef_)

predict_values = regr.predict([[2.79766258, -1.392667]])[0]
print("\nAfter PCA")
print(f"Predicted wage: {round(predict_values)}")
print(f"Different: {round(abs(1-predict_values/721)*100, 2)}%")

## So sánh tỉ lệ trước và sau khi PCA
print(f"\nBefore vs After PCA: {round(abs(1-predict_values/A)*100, 2)}%")