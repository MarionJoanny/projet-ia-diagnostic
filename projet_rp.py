import pandas as pd

file_equip = open("../EQUIPEMENTS.csv", "r")
file_OT = open("../OT_ODR.csv", "r")


df_equipe = pd.read_csv(file_equip, sep=";")
df_OT = pd.read_csv(file_OT, sep=";")


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

