import random

from pca import *
from LinearRegress import LR_PREDICT

X, y = setup_data(feature)
X_train, X_test, y_train, y_test = train_test_data(X, y)
X_pca_train, X_pca_test, y_train, y_test = train_test_data(setup_pca(X), y)

# Số giá trị dự đoán
n = 100
# Độ delay
delay = 0.0
# Ảnh
plot = True
# Diff or Pred
Mode = True
# Detail
Detail = True
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

    y_used = []
    for id in id_list:
        y_used.append(y_test[id])

    print("  Before PCA")
    nor_predict = LR_PREDICT(X_train, X_test, y_train, y_test, id_list)
    nor_predict.gene() if Detail else nor_predict.gene(False)
    nor_avr = nor_predict.get_avr()
    print(f"    Average Diff: {nor_avr}%")

    print()

    print("  After PCA")
    pca_predict = LR_PREDICT(X_pca_train, X_pca_test, y_train, y_test, id_list)
    pca_predict.gene() if Detail else pca_predict.gene(False)
    pca_avr = pca_predict.get_avr()
    print(f"    Average Diff: {pca_avr}%")

    # Độ chênh lệch tỉ lệ trước và sau khi PCA
    diff = round(abs(pca_avr - nor_avr), 1)
    print(f"\nBefore vs After PCA: {diff}%\n")

    with open('csv/.history.csv', 'a') as h:
        h.write(f"{n},{nor_avr},{pca_avr},{diff},{' '.join(feature if loc_feature else '*')}\n")

    if plot:
        if Mode:
            plt.plot(nor_predict.get_all())
            plt.plot(pca_predict.get_all())
            plt.plot([0, n], [nor_avr, nor_avr])
            plt.plot([0, n], [pca_avr, pca_avr])
            plt.legend(['Normal', 'PCA', f'Normal mean ({nor_avr}%)', f'PCA mean ({pca_avr}%)'])
            plt.show()
        else:
            plt.plot(y_used, c='green')
            plt.plot(nor_predict.get_pred())
            plt.plot(pca_predict.get_pred())
            plt.legend(['Test', 'Normal', 'PCA'])
            plt.show()


if __name__ == "__main__":
    if run_main:
        for i in range(0, loop):
            main()
    pass
