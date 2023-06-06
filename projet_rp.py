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

#for i in df_equipe.columns:
    #print(f'################    {i}     #########################')
    #sprint(df_equipe[i].unique())

print('--------------------------------------')

print('----------------OT----------------------')

print(df_OT.describe())
print(df_OT.head())
print(f'isna = {df_OT.isna().sum()}')


#for i in df_OT.columns:
    #print(f'################    {i}     #########################')
    #print(df_OT[i].unique())
print('--------------------------------------')



## cleaning des dataframes

#print(df_OT.head(10))
#print(df_OT[df_OT['KILOMETRAGE'].isna() == True])

df_OT.drop(columns=['KILOMETRAGE'], inplace=True)
#print(df_OT.head(100))


#print(f'moteur {df_equipe["MOTEUR"].unique()}')
#print(df_equipe['EQU_ID'].unique())

#print(df_OT['SIG_ORGANE'].unique())
#print(df_OT['ODR_LIBELLE'].value_counts())
#print(df_OT['TYPE_TRAVAIL'].unique())
#print(df_OT['EQU_ID'].unique())
#print(df_OT['SYSTEM_N1'].value_counts())


## RAPHAEL SPACE HERE
##jointure des dataframes
df_light = df_OT[['SIG_ORGANE', 'SYSTEM_N1', 'EQU_ID']]

df_light = df_light.merge(df_equipe, on='EQU_ID', how='left')

df_light['SIG_ORGANE'] = df_light['SIG_ORGANE'].astype('category')
df_light['SYSTEM_N1'] = df_light['SYSTEM_N1'].astype('category')
df_light['MODELE'] = df_light['MODELE'].astype('category')
df_light['MOTEUR'] = df_light['MOTEUR'].astype('category')

print('####################################################')
df_light.drop(columns=['CONSTRUCTEUR'], inplace=True)
print(df_light.head(10))
print(df_light.columns)

import sklearn.model_selection as tts

x_train, x_test, y_train, y_test = tts.train_test_split(df_light[['MODELE','MOTEUR','SIG_ORGANE']],df_light['SYSTEM_N1'], test_size=0.2, random_state=42)

print(x_train.shape)
print(x_test.shape)


print('####################################################')
## RAPHAEL SPACE HERE

import pyagrum_extra as gum

bn = gum.BayesNet("diag")

mot = bn.add(gum.LabelizedVariable('MOTEUR', 'le moteur', 56))
mdl = bn.add(gum.LabelizedVariable('MODELE', 'le modele', 68))
org = bn.add(gum.LabelizedVariable('SIG_ORGANE', 'l\'organe', len(df_light['SIG_ORGANE'].cat.categories)))
sys = bn.add(gum.LabelizedVariable('SYSTEM_N1', 'localisation', len(df_light['SYSTEM_N1'].cat.categories)))

bn=gum.fastBN("sys<-mot<-mdl->org->sys")

print(bn)
bn.fit_bis(df_light, verbose_mode=True)

