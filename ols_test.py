import statsmodels.api as sm

duncan_prestige = sm.datasets.get_rdataset("Duncan", "carData")
# print(duncan_prestige.data)
Y = duncan_prestige.data['income']
X = duncan_prestige.data['education']
X = sm.add_constant(X)
# print(X)
# print(Y)
model = sm.OLS(Y, X)
# print(model)
result = model.fit()
# print(result.summary())

ypred = result.predict(X)
print(ypred)