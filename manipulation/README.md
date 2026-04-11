# Transfer Learning sur FashionMNIST — TP Complet

## Objectif

Appliquer des modèles pré-entraînés sur ImageNet (AlexNet, ResNet18, VGG16) au dataset FashionMNIST en utilisant trois stratégies de gel des couches, puis comparer les performances.

## Problème à résoudre

| | FashionMNIST | Modèles ImageNet |
|--|--|--|
| Taille | 28×28 | 224×224 |
| Canaux | 1 (gris) | 3 (RGB) |
| Normalisation | — | mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225] |
| Classes | 10 | 1000 |

**Solution :** `Resize(224)` → `Grayscale(3)` → `ToTensor()` → `Normalize(ImageNet)` + remplacement de la dernière couche (1000→10).

## Modèles utilisés

| Modèle | Année | Paramètres | Particularité | Couche modifiée |
|--------|-------|-----------|---------------|-----------------|
| AlexNet | 2012 | ~60M | Premier succès Deep Learning | `classifier[6]` |
| ResNet18 | 2015 | ~11M | Skip connections | `fc` |
| VGG16 | 2014 | ~138M | Excellent extracteur de features | `classifier[6]` |

## Les 9 expériences

### Stratégie 1 — Tout geler sauf la dernière couche

Toutes les couches sont gelées (`requires_grad=False`). Seule la nouvelle couche de sortie est entraînée.

- **Expérience 1 :** AlexNet — `classifier[6]` seul entraîné
- **Expérience 2 :** ResNet18 — `fc` seul entraîné
- **Expérience 3 :** VGG16 — `classifier[6]` seul entraîné

### Stratégie 2 — Gel partiel

Les couches basses (features universelles) restent gelées. Les couches hautes + la dernière couche sont dégelées.

- **Expérience 4 :** AlexNet — `features` gelé, `classifier` entier libre
- **Expérience 5 :** ResNet18 — conv1→layer3 gelés, `layer4` + `fc` libres
- **Expérience 6 :** VGG16 — `features[:24]` gelé, `features[24:]` + `classifier` libres

### Stratégie 3 — Fine-tuning progressif

On commence tout gelé, puis on dégèle couche par couche en réduisant le learning rate à chaque phase (2 epochs par phase).

- **Expérience 7 :** AlexNet — Phase 1: classifier[6] (lr=0.001) → Phase 2: classifier (lr=0.0005) → Phase 3: features[8:] (lr=0.0001)
- **Expérience 8 :** ResNet18 — Phase 1: fc (lr=0.001) → Phase 2: layer4 (lr=0.0005) → Phase 3: layer3 (lr=0.0001)
- **Expérience 9 :** VGG16 — Phase 1: classifier[6] (lr=0.001) → Phase 2: classifier (lr=0.0005) → Phase 3: features[24:] (lr=0.0001)

## Structure du notebook

| Section | Contenu |
|---------|---------|
| 1 | Imports et configuration (device GPU/CPU) |
| 2 | Préparation des données (transforms, DataLoaders) |
| 3 | Fonctions utilitaires (train, evaluate, run_experiment, progressive_finetune, confusion matrix) |
| 4–6 | Expériences 1–3 : gel total |
| 7 | Expériences 4–6 : gel partiel |
| 8 | Expériences 7–9 : fine-tuning progressif |
| 9 | Comparaison : graphiques (accuracy/loss) + tableau récapitulatif des 9 expériences |
| 10 | Évaluation finale : confusion matrix + classification report pour les 3 modèles progressifs |
| 11 | Conclusion |

## Comment exécuter

1. Ouvrir `TransferLearning_TP_COMPLET.ipynb` dans Google Colab
2. Activer le GPU : **Runtime → Change runtime type → T4 GPU**
3. Exécuter les cellules dans l'ordre

**Temps estimé :** ~30–45 min sur T4 GPU pour les 9 expériences.

**Si erreur "CUDA out of memory" :** réduire `BATCH_SIZE` de 32 à 16 dans la cellule 2.

## Résultats attendus

- **Tout gelé :** entraînement rapide, accuracy correcte (~85–90%)
- **Gel partiel :** meilleure accuracy (~88–92%)
- **Progressif :** meilleure accuracy globale (~90–93%)
- La confusion la plus fréquente est entre Shirt/T-shirt/Pullover/Coat (visuellement similaires)

## Dépendances

- PyTorch, torchvision
- scikit-learn
- seaborn, matplotlib, numpy
- tqdm

Toutes sont pré-installées dans Google Colab.
