from sklearn import linear_model
from data import *

data = load_data('cars.csv')
print(data)

X = data[['Weight', 'Volume', 'CO2']].values
y = data['Type'].values

regr = linear_model.LinearRegression()
regr.fit(X, y)

# Dự đoán Type dự trên Weight và Volume và CO2
predType = regr.predict([[900, 865, 90]])

# Giá trị được dự đoán
print(round(predType[0]))
# Hệ số của giá trị dự đoán
print(regr.coef_)