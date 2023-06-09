# projet-ia-diagnostic
Création d'un réseau bayésien pour la génération de diagnostics et recommandations d'actions de maintenance

* Etape 1 : Télécharger le repo et le dézipper

* Etape 2 : Placer les fichiers de données directement dans le dossier du repo. 

* Etape 3 : Lancer le code avec les commandes indiquées ci-dessous : 


MODELE 1 qui prédit SYSTEM_N1 à partir de la variable SIG_ORGANE, MODELE, MOTEUR. 
Nous avons ensuite à partir de ce modèle 1 remplacer SYSTEM_N1 par SYSTEM_N2 et SYSTEM_N3.
```
python3 diag_model1.py
``` 

MODELE 2 qui prédit SYSTEM_N3 à partir de la variable SIG_ORGANE, MODELE, MOTEUR, SYSTEM_N1 et SYSTEM_N2. Nous avons également testé d'enlever SYSTEM_N3 et de ne garder que SYSTEM_N2 (elle devient alors la variable cible et plus une variable explicative). 

Puis pour le modèle 2 la commande ci-dessous : 

```
python3 diag_model2.py
```
