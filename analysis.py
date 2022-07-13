import pandas as pd

his = pd.read_csv('.history.csv')

his = his[his['Before'] <= 5]

print(his.loc[:, his.columns != 'Feature'])
print(his)