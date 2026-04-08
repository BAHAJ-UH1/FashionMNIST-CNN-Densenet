# Classification d'Images Fashion MNIST

## Aperçu

Ce projet utilise le jeu de données Fashion MNIST, un jeu de données populaire composé d'images d'articles de Zalando.

Créé comme un remplaçant moderne et direct du jeu de données original de chiffres manuscrits MNIST, Fashion MNIST pose un problème de vision par ordinateur légèrement plus difficile. Alors que le MNIST standard est souvent considéré comme trop facile pour les réseaux de neurones modernes (qui peuvent facilement atteindre une précision >99%), Fashion MNIST offre plus de complexité et de variance, ce qui en fait un excellent point de référence pour tester et valider les algorithmes d'apprentissage automatique.

## Détails du Jeu de Données

Le jeu de données se compose de 70 000 images en niveaux de gris au total, réparties en :

Ensemble d'entraînement : 60 000 images

Ensemble de test : 10 000 images

Format de l'image : * 28 x 28 pixels

Niveaux de gris (les valeurs des pixels vont de 0 à 255, où les nombres plus élevés représentent des pixels plus sombres)

784 pixels au total par image

## Installation des Dépendances

Vous pouvez installer les paquets requis directement en utilisant pip. Il est fortement recommandé de le faire dans un environnement virtuel (comme venv ou conda).

```bash
# 1. Créer et activer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows utilisez : venv\Scripts\activate

# 2. Installer le projet et ses dépendances
pip install .
```