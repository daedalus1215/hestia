import pandas as pd

df = pd.read_excel('../resources/fake_news.xlsx')
print('df.head()\n')
print(df.head())
print('df.isnull\n')
print(df.isnull().sum())
print('df drops nulls\n')
df = df.dropna() # remove nulls
print('df.isnull\n')
print(df.isnull().sum())
print('shape \n')
print(df.shape)
print('label counts \n')
print(df['label'].value_counts())
