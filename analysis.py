import pandas as pd

his = pd.read_csv('csv/.history.csv')

print(f"Mean before: {round(his['Before'].mean(), 1)}%")
print(f"Mean after: {round(his['After'].mean(), 1)}%")

print()
his = his[his['After'] <= 25]
his = his[his['Before'] <= 6]

print(his.loc[:, his.columns != 'Feature'])
print()
print(his['Feature'])