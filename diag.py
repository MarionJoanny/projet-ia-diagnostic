import pandas as pd
import os
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
from pyagrum_extra import gum
#import ipdb
import sklearn.model_selection as tts

# Modélisation
# ============

# Chargement et préparation des données
ot_odr_filename = os.path.join(".", "../OT_ODR.csv")
ot_odr_df = pd.read_csv(ot_odr_filename,
                        sep=";")
print(ot_odr_df.SIG_ORGANE.isna().sum())

equipements_filename = os.path.join(".", "../EQUIPEMENTS.csv")
equipements_df = pd.read_csv(equipements_filename,
                        sep=";")
                        
data_df = ot_odr_df.merge(equipements_df,how='right',on='EQU_ID')
print(data_df.head(5))

var_cat = ['ODR_LIBELLE', 'TYPE_TRAVAIL',
           'SYSTEM_N1', 'SYSTEM_N2', 'SYSTEM_N3', 
           'SIG_ORGANE', 'SIG_CONTEXTE', 'SIG_OBS', 'LIGNE','MODELE','MOTEUR']
           
for var in var_cat:
    data_df[var] = data_df[var].astype('category')

print(data_df.SIG_ORGANE.isna().sum())

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
print(x_train.head(2))
bn.fit_bis(x_train, verbose_mode=True)

##############
# EVALUATION #
##############
print(x_test.SIG_ORGANE.isna().sum())
y_pred = bn.predict(x_test,'SYSTEM_N1')
n = len(y_pred)
acc = 0
for k in range(n):
    if y_pred[k] == y_test[k]:
        acc += 1
acc /= n
print('accuracy : %s%', str(acc*100))

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
             style={'width': '65%', 'float': 'right', 'display': 'inline-block'})
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


if __name__ == '__main__':
    app.run_server(debug=True)
