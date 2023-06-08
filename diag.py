import pandas as pd
import os
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
from pyagrum_extra import gum
import numpy as np
#import ipdb
import sklearn.model_selection as tts
from sklearn.metrics import confusion_matrix

# Modélisation
# ============

# Chargement et préparation des données
ot_odr_filename = os.path.join(".", "../OT_ODR.csv")
#ot_odr_filename = os.path.join("OT_ODR.csv")
ot_odr_df = pd.read_csv(ot_odr_filename,
                        sep=";")
                        
equipements_filename = os.path.join(".", "../EQUIPEMENTS.csv")
#equipements_filename = os.path.join("EQUIPEMENTS.csv")
equipements_df = pd.read_csv(equipements_filename,
                        sep=";")
                        
df_sub_OT = ot_odr_df[['SIG_ORGANE', 'SYSTEM_N1', 'EQU_ID']]

data_df = df_sub_OT.merge(equipements_df, on='EQU_ID', how='left')
                        
var_cat = ['SYSTEM_N1', 'SIG_ORGANE', 'EQU_ID','MODELE','MOTEUR']
           
for var in var_cat:
    data_df[var] = data_df[var].astype('category')


#########################
# SEPARATION TEST TRAIN #
#########################
x_train, x_test, y_train, y_test = tts.train_test_split(data_df[['MODELE','MOTEUR','SIG_ORGANE']],data_df['SYSTEM_N1'], test_size=0.2, random_state=42)

print(x_train.shape)
print(x_test.shape)

# Configuration du modèle
model_name = "Outil de diagnostic"
var_features = ["SIG_ORGANE", "MODELE","MOTEUR"] # Variables explicatives
var_targets = ["SYSTEM_N1"] # Variables à expliquer
arcs = [("MODELE", "SIG_ORGANE"),
        ("MODELE", "MOTEUR"),
        ("SYSTEM_N1", "SIG_ORGANE"),
        ("SYSTEM_N1", "MOTEUR")]

# Création du modèle
var_to_model = var_features + var_targets
var_bn = {}
for var in var_to_model:
    nb_values = len(data_df[var].cat.categories)
    var_bn[var] = gum.LabelizedVariable(var, var, nb_values)

for var in var_bn:
    for i, modalite in enumerate(data_df[var].cat.categories):
        var_bn[var].changeLabel(i, modalite)

bn = gum.BayesNet(model_name)

for var in var_bn.values():
    bn.add(var)

for arc in arcs:
    bn.addArc(*arc)

# Apprentissage des LPC

x_train['SYSTEM_N1'] = y_train

bn.fit_bis(x_train, verbose_mode=True)

##############
# EVALUATION #
##############

# y_pred = bn.predict(x_test,'SYSTEM_N1')
# n = len(y_pred)

# unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
# print(unique_pred)
# print(counts_pred)
# print("****************************************")
# unique_test, counts_test = np.unique(y_test, return_counts=True)
# print(unique_test)
# print(counts_test)

# acc = (y_pred==y_test).sum()
# acc /= n
# print('accuracy : {}'.format(str(acc*100)))

# print(confusion_matrix(y_test, y_pred,labels=['DIVERS','EQUIPEMENT CHASSIS','EQUIPEMENT CLIMATIQUE',
#  'EQUIPEMENT DE CARROSSERIE','EQUIPEMENT DE FREINAGE',
#  'EQUIPEMENT DE MOTORISATION','EQUIPEMENT DE TRANSMISSION',
#  'EQUIPEMENT ELECTRIQUE','EQUIPEMENT EMBARQUE','EQUIPEMENT PNEUMATIQUE']))

# Création de l'application
# =========================
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1(model_name),
    html.Div([
        html.Div([
            html.Label(var),
            dcc.Dropdown(
                id=f'{var}-dropdown',
                options=[{'label': i, 'value': i} for i in data_df[var].cat.categories],
                value=data_df[var].cat.categories[0]
            )
        ]) for var in var_features],
        style={'width': '30%', 'display': 'inline-block'}),
    html.Div([
            dcc.Graph(id=f'{var}-graph') 
        for var in var_targets],
             style={'width': '65%', 'float': 'right', 'display': 'inline-block'}),
    html.Div([
        html.Div(id=f'{i}-div') for i in range(5)
    ])
])


@app.callback(
    [Output(f'{var}-graph', 'figure') for var in var_targets],
    [Input(f'{var}-dropdown', 'value') for var in var_features]
)
def update_graph(*var_features_values):
    bn_ie = gum.LazyPropagation(bn)

    ev = {var: value for var, value in zip(var_features, var_features_values)}
    bn_ie.setEvidence(ev)
    bn_ie.makeInference()

    prob_target = []
    for var in var_targets:
        prob_target_var = bn_ie.posterior(var).topandas().droplevel(0)
        prob_fig = px.bar(prob_target_var)
        prob_target.append(prob_fig)
        
    return tuple(prob_target)

@app.callback(
    [Output(f'{i}-div', 'children') for i in range(5)],
    [Input(f'{var}-dropdown', 'value') for var in var_features]
)
def update_text(*var_features_values):
    bn_ie = gum.LazyPropagation(bn)

    ev = {var: value for var, value in zip(var_features, var_features_values)}
    bn_ie.setEvidence(ev)
    bn_ie.makeInference()

    for var in var_targets:
        prob_target_var = bn_ie.posterior(var).topandas()

    pretty_string = ''
    for column,val in prob_target_var.sort_values().items():
        pretty_string = str(column[1]) + '\t\t' + "{0:.0%}".format(val) + '\r\n' + pretty_string
        
    return pretty_string.split('\r\n')[:5]

if __name__ == '__main__':
    app.run_server(debug=True)
