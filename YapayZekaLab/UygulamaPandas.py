import pandas as pd

#pandas single type data declaration
s = pd.Series([12,56,23,21], index = ['a','b','c','d'])

#reading  data from an excel file
data = pd.read_excel('deneme.xlsx')
#chosing data from table location
value = data.iloc[1,3]
#deleting row and column from table
delRow = data.drop([0])
delCol = data.drop(['Nem'], axis=1) #axis=1 column, axis=0 row

print(delRow)