from data import *

filename = "Wage_data.csv"
raw_data = load_data(filename)
pca_data = pca_data(filename)

print(raw_data.values)
print(pca_data)
