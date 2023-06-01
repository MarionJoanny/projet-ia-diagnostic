import pandas as pd

file_equip = open("../EQUIPEMENTS.csv", "r")
file_OT = open("../OT_ODR.csv", "r")


df_equipe = pd.read_csv(file_equip, sep=";")
df_OT = pd.read_csv(file_OT, sep=";")


print(df_equipe.describe())
print(df_equipe.head())
print('--------------------------------------')

print(df_OT.describe())
print(df_OT.head())
print('--------------------------------------')

