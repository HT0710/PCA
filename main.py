from data import *

filename = "Wage_data.csv"
dataset = load_data(filename)
dataset = dataset.loc[:, ['nearc4', 'educ', 'age', 'black', 'wage', 'IQ', 'married', 'exper']]

X = dataset.loc[:, dataset.columns != 'wage'].values
X_pca = pca_data(X)
y = dataset['wage'].values
LR = LinearRegression()


class LR_PREDICT():
    def __init__(self, X, y, id_list):
        self.X = X
        self.y = y
        self.id_list = id_list
        self.avr = 0

    def get_avr(self):
        return self.avr

    def gene(self, amount):
        LR.fit(self.X, self.y)
        i = 0
        for id in self.id_list:
            pred = round(LR.predict([self.X[id].tolist()])[0])
            diff = round(abs(1 - (pred / self.y[id])) * 100, 1)
            self.avr += diff

            print(f"{i} | Predicted wage: {pred} | Diff: {diff}%")
            i += 1
        print('-'*40)
        self.avr = round(self.avr / len(self.id_list), 1)


n = 6


id_list = []
for i in range(n):
    id_list.append(random.randint(0, X.shape[0]))

print("  Before PCA")
nor_predict = LR_PREDICT(X, y, id_list)
nor_predict.gene(n)
nor_avr = nor_predict.get_avr()
print(f"    Average Diff: {nor_avr}%")

print()

print("  After PCA")
pca_predict = LR_PREDICT(X_pca, y, id_list)
pca_predict.gene(n)
pca_avr = pca_predict.get_avr()
print(f"    Average Diff: {pca_avr}%")

## So sánh tỉ lệ trước và sau khi PCA
print(f"\nBefore vs After PCA: {round(abs(pca_avr - nor_avr), 1)}%")
