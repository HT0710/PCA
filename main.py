from data import *

filename = "Wage_data.csv"
raw_data = load_data(filename)
raw_data = raw_data.loc[:,
           ['nearc4', 'educ', 'age', 'black', 'wage', 'IQ', 'married', 'exper']]

pca_data = pca_data(raw_data)

X = raw_data[['nearc4', 'educ', 'age', 'black', 'IQ', 'married', 'exper']]
y = raw_data['wage']

# test với 5 giá trị đầu
X, y = X.head().values, y.head().values

regr = linear_model.LinearRegression()
regr.fit(X, y)


def predict(nearc4, educ, age, black, iq, married, exper):
    value = regr.predict([[nearc4, educ, age, black, iq, married, exper]])
    return round(value[0])


print(raw_data[:1])
# dự đoán giá trị = với id 0
print(f"\nPredicted wage: {predict(0, 12, 27, 0, 93, 1, 9)}")
# kết quả = với id 0 => dự đoán chính xác

print(f"\nPredicted wage: {predict(1, 16, 25, 0, 100, 0, 12)}")
