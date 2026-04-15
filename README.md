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

Ce projet utilise [uv](https://github.com/astral-sh/uv), un gestionnaire de paquets et d'environnements Python extrêmement rapide écrit en Rust.

### Option 1 : Utilisation de `uv` (Recommandé)

Si vous n'avez pas encore `uv`, installez-le via :
```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
```
Ensuite, configurez le projet en une seule commande :

# Crée l'environnement virtuel et installe les dépendances
uv venv
source .venv/bin/activate  # Sur Windows : .venv\Scripts\activate
uv pip install -e .


