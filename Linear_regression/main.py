from sklearn import linear_model
from data import *

data = load_data('cars.csv')

X = data[['Weight', 'Volume']].values
y = data['CO2'].values

regr = linear_model.LinearRegression()
regr.fit(X, y)

# Dự đoán giá trị Co2 dự trên 3300 Weight và 1300 Volume
predictedCO2 = regr.predict([[3300, 1300]])

# Giá trị được dự đoán
print(predictedCO2)
# Hệ số của giá trị dự đoán so với Weight và Volume
print(regr.coef_)