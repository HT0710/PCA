from sklearn.linear_model import LinearRegression

LR = LinearRegression()


class LR_PREDICT:
    def __init__(self, X_train, X_test, y_train, y_test, id_list):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.id_list = id_list
        self.avr = 0
        self.fred = []
        self.diff = []

    def get_avr(self):
        return self.avr

    def get_pred(self):
        return self.fred

    def get_all(self):
        return self.diff

    def gene(self, prnt: bool = True):
        LR.fit(self.X_train, self.y_train)
        i = 0
        for id in self.id_list:
            pred = LR.predict([self.X_test[id].tolist()]).round()[0]
            diff = round(abs(1 - (pred / self.y_test[id])) * 100, 1)
            self.fred.append(pred)
            self.diff.append(diff)
            self.avr += diff

            if prnt:
                print(f"{i} | Predicted: {pred} | Diff: {diff}%")
            i += 1
        print('-' * 40)
        self.avr = round(self.avr / len(self.id_list), 1)