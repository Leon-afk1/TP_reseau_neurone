# Mini-Projet : Classification de Panneaux et Localisation de Formes

Ce dépôt contient le code source et les résultats du mini-projet sur les réseaux de neurones. L'objectif est double : classer des panneaux de signalisation réels (base de données GTSDB) et localiser des formes géométriques dans des images synthétiques.

## Contenu du projet

Le projet est divisé en 6 parties distinctes traitées dans le Notebook :

- Exploration et Préparation : Extraction des panneaux depuis les images GTSDB, redimensionnement (32x32) et création des jeux d'Entraînement/Validation/Test.

- MLP (Perceptron Multicouche) : Classification "naïve" avec un réseau dense.

- CNN (Réseau Convolutif) : Classification avec une architecture spécialisée (Conv2D + MaxPooling).

- Transfer Learning (VGG16) : Adaptation d'un réseau pré-entraîné (VGG16) via Full Fine-Tuning.

- Localisation (Régression) : Détection de coordonnées (Bounding Box) sur des formes générées (Carré, Cercle, Triangle) avec variations (translation, échelle, rotation).

- Synthèse : Comparaison globale des approches.

## Pré-requis et Installation

Pour exécuter le notebook, il faut 

1. Créer un environnement virtuel :

```bash
python3 -m venv venv
```

2. Activer l'environnement virtuel créé :

```bash
source venv/bin/activate # Sur Linux/Mac
venv\Scripts\activate # Sur Windows
```

3. Installer les dépendances :

```bash
pip install -r requirements.txt
```

## Utilisation

### Données GTSDB

1. Télécharger le dataset à l’adresse suivante : https://erda.ku.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/TrainIJCNN2013.zip.

2. Décompresser le dataset à la racine du projet

3. Éventuellement, mettre à jour la valeur de la variable `DATA_PATH` (Partie 1 - Question 1.a)

### Exécution

Exécuter les cellules du notebook `Notebook.ipynb`.

### Sauvegarde

À la fin de l'exécution, les modèles entraînés sont sauvegardés au format `.keras` :
- `mlp_traffic_signs.keras.keras`
- `cnn_traffic_signs.keras`
- `vgg16_finetuned_traffic_signs.keras`
- `cnn_localization.keras`

## Résumé des Résultats

Voici les performances obtenues sur le jeu de test indépendant :

### Partie Classification (GTSDB)

| Modèle             | Précision (Accuracy) | Observations                                                                                   |
| ------------------ | -------------------- | ---------------------------------------------------------------------------------------------- |
| MLP                | ~86%                 | Sur-apprentissage marqué, sensible aux translations.                                           |
| CNN (Custom)       | ~92%                 | Excellent compromis poids/performance pour images 32x32.                                       |
| VGG16 (Fine-Tuned) | ~93%                 | Meilleure stabilité. Nécessite le déblocage des poids et un pré-traitement spécifique (0-255). |


### Partie Localisation (Formes Géométriques)

L'invariance est réussie à la translation, l'échelle et la rotation.

On obtient une erreur moyenne absolue (MAE) de 0.31 pixels (Précision sub-pixel).

Toutefois, l'architecture échoue à détecter plusieurs objets simultanément (problème structurel résoluble par YOLO/SSD).

## Auteurs

- BASKAR Arnold
- MORALES Léon
- NARESH PRABAHARAN Vinith