import pandas as pd

his = pd.read_csv('.history.csv')

his = his[his['After'] <= 20]
his = his[his['Before'] <= 6]

print(his.loc[:, his.columns != 'Feature'])
print(his)