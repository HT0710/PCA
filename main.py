import random
import time
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from data import *
from pca import train_test_pca

X, y = setup_data("csv/_Wage_data.csv")
X_train, X_test, y_train, y_test = train_test_data(X, y)
X_pca_train, X_pca_test, y_train, y_test = train_test_pca(X, y)
LR = LinearRegression()

# Số giá trị dự đoán
n = 100
# Độ delay
delay = 0.001
# Ảnh
plot = True
# Số lần dự đoán
loop = 1
"""Lưu ý
!!! loop = 10 nghĩa là lặp lại 10 lần dự đoán n !!!
Tổng số lần dự đoán = n * loop
Mỗi lần loop đều sẽ được lưu vào history
"""

# Thực thi main
run_main = True


def main():
    id_list = []
    for i in range(n):
        id_list.append(random.randint(0, X_test.shape[0] - 1))

    print("  Before PCA")
    nor_predict = LR_PREDICT(X_train, X_test, y_train, y_test, id_list)
    nor_predict.gene()
    nor_avr = nor_predict.get_avr()
    print(f"    Average Diff: {nor_avr}%")

    print()

    print("  After PCA")
    pca_predict = LR_PREDICT(X_pca_train, X_pca_test, y_train, y_test, id_list)
    pca_predict.gene()
    pca_avr = pca_predict.get_avr()
    print(f"    Average Diff: {pca_avr}%")

    # Độ chênh lệch tỉ lệ trước và sau khi PCA
    diff = round(abs(pca_avr - nor_avr), 1)
    print(f"\nBefore vs After PCA: {diff}%\n")

    with open('csv/.history.csv', 'a') as h:
        h.write(f"{n},{nor_avr},{pca_avr},{diff},{' '.join(feature if loc_feature else '*')}\n")

    if plot:
        plt.plot(nor_predict.get_all())
        plt.plot(pca_predict.get_all())
        plt.plot([0, n], [nor_avr, nor_avr])
        plt.plot([0, n], [pca_avr, pca_avr])
        plt.legend(['Normal', 'PCA', f'Normal mean ({nor_avr}%)', f'PCA mean ({pca_avr}%)'])
        plt.show()


# train model bằng tập train sau đó dự đoán = tập test
class LR_PREDICT:
    def __init__(self, X_train, X_test, y_train, y_test, id_list):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.id_list = id_list
        self.avr = 0
        self.all = []

    def get_avr(self):
        return self.avr

    def get_all(self):
        return self.all

    def gene(self):
        LR.fit(self.X_train, self.y_train)
        i = 0
        for id in self.id_list:
            time.sleep(delay)
            pred = LR.predict([self.X_test[id].tolist()]).round()[0]
            diff = round(abs(1 - (pred / self.y_test[id])) * 100, 1)
            self.all.append(diff)
            self.avr += diff

            print(f"{i} | Predicted: {pred} | Diff: {diff}%")
            i += 1
        print('-' * 40)
        self.avr = round(self.avr / len(self.id_list), 1)


if __name__ == "__main__":
    if run_main:
        for i in range(0, loop):
            main()
    pass
