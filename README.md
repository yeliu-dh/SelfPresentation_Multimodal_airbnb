# Stratégies de la présentation de soi des hôtes Airbnb en réponse aux Jeux Olympiques de Paris 2024

## 📌 Présentation du projet

Ce projet analyse la dynamique des stratégies de présentation de soi sur Airbnb en prenant les Jeux Olympiques de Paris 2024 comme un choc exogène affectant le marché de la location de courte durée.

Dans un contexte de plateforme P2P, où la confiance joue un rôle essentiel, les hôtes Airbnb mobilisent non seulement le système de réputation, mais aussi leur présentation de soi afin d'envoyer des signaux de crédibilité et de réduire l'incertitude.

## 🙋‍♀️ Questions de recherche

1. Comment les hôtes ajustent-ils leur présentation de soi ?

2. Existe-il une logique similaire entre les signaux textuels et visuels ?

3. Les nouveaux entrants et les hôtes établis, ajustent-ils leur présentation de soi de manière différente ?

## 📊 Données

Cette étude s’appuie sur des données Airbnb collectées à Paris et à Londres à trois périodes sur le site [Inside Airbnb](https://insideairbnb.com/get-the-data/):

- Juin 2023
- Décembre 2023
- Juin 2024

## 🧑‍💼 Définition des hôtes

Les hôtes sont classés en deux groupes :  
- **nouveaux hôtes** : hôtes dont l’identifiant n’apparaît pas dans la vague d’observation précédente ;  
- **hôtes établis** : hôtes déjà présents dans les vagues précédentes.

---

## 🧠 Construction des tactiques de présentation de soi

### Tactiques textuelles

Cinq dimensions sont identifiées :
- ouverture
- authenticité
- sociabilité
- auto-promotion
- exemplarité

### Tactiques visuelles

Trois catégories de photos :
- portrait
- photo de vie
- absence de personne


## 🔬 Méthodologie et pipeline d’analyse

Les analyses s’appuient sur un pipeline en trois étapes :

### 1. Construction des variables de présentation de soi

Les tactiques textuelles sont construites à partir d’une échelle de mesure validée par analyse factorielle exploratoire, puis quantifiées à l’aide d’un modèle de classification zero-shot (MoritzLaurer/bge-m3-zeroshot-v2.0) appliqué aux descriptions des hôtes.

### 2. Analyse des contenus visuels

Les photos de profil sont traitées à l’aide d’un pipeline combinant :
- détection faciale (MTCNN),
- similarité image-texte (CLIP),

### 3. Estimation économétrique

Les effets du choc olympique sont analysés à l’aide de modèles en différences-en-différences (OLS), afin d’étudier :
- l’effet du choc sur l’ensemble des hôtes,
- l'effet hétérogène selon le profil d’hôte.

---
## 📈 Résultats principaux 
### Descriptifs
Au niveau global, les résultats descriptifs montrent une prédominance des tactiques promotionnelles sur le plan textuel, tandis que les dimensions visuelles apparaissent globalement équilibrées.

En termes d’hétérogénéité, les nouveaux hôtes présentent des profils textuels plus homogènes, avec des valeurs proches de zéro, tandis que les hôtes établis affichent des orientations plus marquées, notamment vers l’auto-promotion.

Sur le plan visuel, aucune différence notable n’est observée à Londres entre les deux groupes, tandis qu’à Paris, les nouveaux hôtes se distinguent par une utilisation plus fréquente des photos de type portrait.

### DID
Les estimations en différences-en-différences indiquent que la majorité des tactiques évoluent plus favorablement à Paris qu’à Londres.

Ces ajustements concernent notamment les stratégies promotionnelles et les photos incluant une présence humaine, qui suggèrent une intensification des signaux de confiance directes et rapides dans un contexte de concurrence accrue.

Des effets d’hétérogénéité apparaissent selon le profil des hôtes : les nouveaux entrants présentent des ajustements plus larges et diversifiés, tandis que les hôtes établis adoptent des stratégies plus ciblées et cohérentes avec leurs pratiques existantes.

---

## 📂 Structure du dépôt

- **A_preprocess/** : prétraitement des données Airbnb  (nettoyage, filtrage)
- **B_images/** : tactiques visuelles  (détection faciale + clip)
- **C_bio/** : tactiques textuelles (zéro-shot + fa)  
- **D_stats/** : analyses économétriques (DID)  
- **utils/** : fonctions utilitaires  
- **requirements.txt** : dépendances  
- **README.md** : documentation

ps. Les données ne sont pas incluses dans le dépôt pour des raisons de volume et de diffusion, mais les scripts nécessaires à la reproduction complète de l’analyse sont fournis. Les données peuvent être obtenues sur demande auprès de l’auteur.


## ⚙️ Reproduction

1. Python version:
```bash
Python 3.10
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt

