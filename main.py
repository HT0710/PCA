from data import *

filename = "Wage_data.csv"
raw_data = load_data(filename).to_numpy()
pca_data = pca_data(filename)

print(raw_data.shape)
print(pca_data.shape)
