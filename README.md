# ML_Project

Collection de projets de Machine Learning explorant diverses techniques d'apprentissage automatique, du preprocessing à la mise en production, avec un focus sur les applications pratiques et l'analyse de données.

## 📋 Description

Ce dépôt regroupe plusieurs projets de machine learning couvrant différents domaines : classification, régression, clustering, et analyse de données. Chaque projet démontre l'application de techniques ML sur des datasets réels avec une approche méthodologique complète.

## 🎯 Objectifs

- **Exploration** : Analyse exploratoire de différents types de données
- **Modélisation** : Implémentation d'algorithmes ML variés
- **Évaluation** : Métriques de performance et validation croisée
- **Déploiement** : Mise en production des modèles
- **Optimisation** : Tuning d'hyperparamètres et amélioration des performances

## 📂 Structure des projets

### Classification
- **Analyse de sentiment** : Classification de textes avec NLP

### Régression
- **Prédiction de prix** : Estimation de valeurs immobilières
- 
### Clustering
- **Segmentation client** : Groupement comportemental

## 🛠️ Technologies utilisées

### Langages et frameworks
- **Python 3.8+** - Langage principal
- **Jupyter Notebook** - Environnement d'expérimentation
- **Pandas** - Manipulation de données
- **NumPy** - Calculs numériques
- **Matplotlib/Seaborn** - Visualisations

### Machine Learning
- **Scikit-learn** - Algorithmes ML classiques
- **NLTK/spaCy** - Traitement du langage naturel


## 🚀 Projets inclus

### 1. Classification de Sentiment
```
sentiment_analysis/
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── evaluation.py
└── models/
    └── sentiment_model.pkl
```

**Objectif** : Classifier des avis clients comme positifs/négatifs
**Techniques** : TF-IDF, Word2Vec, LSTM
**Performance** : Accuracy > 92%


## 📊 Méthodes et techniques

### Preprocessing
- **Nettoyage des données** : Gestion des valeurs manquantes
- **Feature Engineering** : Création de variables dérivées
- **Normalisation** : StandardScaler, MinMaxScaler
- **Encodage** : LabelEncoder, OneHotEncoder, TFiDF

### Modélisation
- **Algorithmes non-supervisés** : K-Means
-

### Évaluation
- **Métriques classification** : Accuracy, Precision, Recall, F1-Score
- **Métriques régression** : RMSE, MAE, R²
- **Validation croisée** : K-Fold, Stratified K-Fold
- **Courbes** : ROC, Precision-Recall


### Visualisations

Chaque projet inclut :
- **Analyse exploratoire** avec graphiques descriptifs
- **Matrices de confusion** pour la classification
- **Courbes d'apprentissage** pour le suivi de l'entraînement
- **Feature importance** pour l'interprétabilité

