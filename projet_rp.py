import pandas as pd

file_equip = open("../EQUIPEMENTS.csv", "r")
file_OT = open("../OT_ODR.csv", "r")


df_equipe = pd.read_csv(file_equip, sep=";")
df_OT = pd.read_csv(file_OT, sep=";")

## description des dataframes

print('-------------equipe-------------------------')


print(df_equipe.describe())
print(df_equipe.head())
print(f'isna = {df_equipe.isna().sum()}')
for i in df_equipe.columns:
    print(f'################    {i}     #########################')
    print(df_equipe[i].unique())

print('--------------------------------------')

print('----------------OT----------------------')

print(df_OT.describe())
print(df_OT.head())
print(f'isna = {df_OT.isna().sum()}')


for i in df_OT.columns:
    print(f'################    {i}     #########################')
    print(df_OT[i].unique())
print('--------------------------------------')



## cleaning des dataframes

print(df_OT.head(10))
print(df_OT[df_OT['KILOMETRAGE'].isna() == True])

df_OT.drop(columns=['KILOMETRAGE'], inplace=True)
print(df_OT.head(100))