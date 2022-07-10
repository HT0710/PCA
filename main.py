from data import *

filename = "_Wage_data.csv"
dataset = load_data(filename)
# nearc4,educ,age,weight,step14,black,wage,enroll,KWW,IQ,married,libcrd14,exper,lwage,expersq
feature = ['nearc4', 'educ', 'age', 'black', 'wage', 'IQ', 'exper', 'lwage']
dataset = dataset.loc[:, feature]

X = dataset.loc[:, dataset.columns != 'wage'].values
X_pca = pca_data(X)
y = dataset['wage'].values
LR = LinearRegression()

# his = load_data('.history.csv')
# print(his)


# Số lượng dự đoán
n = 5


def main():
    id_list = []
    for i in range(n):
        id_list.append(random.randint(0, X.shape[0]))

    print("  Before PCA")
    nor_predict = LR_PREDICT(X, y, id_list)
    nor_predict.gene()
    nor_avr = nor_predict.get_avr()
    print(f"    Average Diff: {nor_avr}%")

    print()

    print("  After PCA")
    pca_predict = LR_PREDICT(X_pca, y, id_list)
    pca_predict.gene()
    pca_avr = pca_predict.get_avr()
    print(f"    Average Diff: {pca_avr}%")

    # Độ chênh lệch tỉ lệ trước và sau khi PCA
    diff = round(abs(pca_avr - nor_avr), 1)
    print(f"\nBefore vs After PCA: {diff}%\n")

    with open('.history.csv', 'a') as h:
        h.write(f"{n},{nor_avr},{pca_avr},{diff},{' '.join(feature)}\n")



class LR_PREDICT():
    def __init__(self, X, y, id_list):
        self.X = X
        self.y = y
        self.id_list = id_list
        self.avr = 0

    def get_avr(self):
        return self.avr

    def gene(self):
        LR.fit(self.X, self.y)
        i = 0
        for id in self.id_list:
            pred = round(LR.predict([self.X[id].tolist()])[0])
            diff = round(abs(1 - (pred / self.y[id])) * 100, 1)
            self.avr += diff

            print(f"{i} | Predicted wage: {pred} | Diff: {diff}%")
            i += 1
        print('-' * 40)
        self.avr = round(self.avr / len(self.id_list), 1)


if __name__ == "__main__":
    main()
    pass
