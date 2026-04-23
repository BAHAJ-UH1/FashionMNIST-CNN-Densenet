# 👗 FashionMNIST — CNN & DenseNet Classifier

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Comparaison de deux architectures deep learning sur le dataset **FashionMNIST** :
un **CNN personnalisé** et un **DenseNet** entraînés et évalués avec les meilleures pratiques.

---

## 📊 Résultats

| Modèle | Accuracy (Test) | Paramètres |
|---|---|---|
| CNN | ~92% | ~450K |
| DenseNet | ~93.5% | ~380K |

---

## 🗂️ Structure du projet

```
FashionMNIST-CNN-Densenet/
├── FashionMNIST-CNN-Densenet.ipynb   # Notebook principal
├── README.md                          # Ce fichier
├── data/                              # Téléchargé automatiquement
├── cnn_fashionmnist.pth               # Poids CNN sauvegardés
└── densenet_fashionmnist.pth          # Poids DenseNet sauvegardés
```

---

## 🚀 Lancement rapide

```bash
# Cloner le repo
git clone https://github.com/BAHAJ-UH1/FashionMNIST-CNN-Densenet.git
cd FashionMNIST-CNN-Densenet

# Installer les dépendances
pip install torch torchvision matplotlib seaborn scikit-learn

# Lancer Jupyter
jupyter notebook FashionMNIST-CNN-Densenet.ipynb
```

---

## 🏗️ Architectures

### CNN Personnalisé
```
Input (1×28×28)
  → Conv(32) → BN → ReLU → Conv(32) → BN → ReLU → MaxPool → Dropout
  → Conv(64) → BN → ReLU → Conv(64) → BN → ReLU → MaxPool → Dropout
  → Conv(128) → BN → ReLU → AdaptiveAvgPool(3×3)
  → FC(1152→256) → ReLU → Dropout
  → FC(256→128) → ReLU → Dropout
  → FC(128→10)
```

### DenseNet
```
Input (1×28×28)
  → Stem Conv(32)
  → DenseBlock(4 layers, k=16) → TransitionLayer(compression=0.5)
  → DenseBlock(4 layers, k=16) → TransitionLayer(compression=0.5)
  → DenseBlock(4 layers, k=16)
  → GlobalAvgPool → Dropout → FC(10)
```

---

## ⚙️ Fonctionnalités

| Feature | Description |
|---|---|
| ✅ Data Augmentation | RandomHorizontalFlip, RandomCrop, ColorJitter |
| ✅ LR Scheduler | CosineAnnealingLR |
| ✅ Early Stopping | Arrêt automatique (patience=7) |
| ✅ Mixed Precision | AMP automatique sur GPU |
| ✅ Label Smoothing | CrossEntropyLoss(label_smoothing=0.05) |
| ✅ Initialisation des poids | Kaiming / Xavier |
| ✅ Reproductibilité | Seed fixe (42) sur tous les modules |

---

## 📈 Visualisations générées

- `dataset_preview.png` — Exemples d'images par classe
- `training_curves.png` — Courbes Loss & Accuracy (CNN vs DenseNet)
- `confusion_matrices.png` — Matrices de confusion comparées
- `error_examples.png` — Exemples de mauvaises prédictions
- `per_class_accuracy.png` — Accuracy par classe

---

## 📦 Dépendances

```
torch >= 2.0
torchvision >= 0.15
numpy
matplotlib
seaborn
scikit-learn
```

---

## 🙏 Contribuer

Les contributions sont les bienvenues ! Voir le [guide de contribution](CONTRIBUTING.md).

1. Forker le repo
2. Créer une branche (`git checkout -b feature/amélioration`)
3. Committer (`git commit -m 'feat: description claire'`)
4. Pusher (`git push origin feature/amélioration`)
5. Ouvrir une Pull Request

---

## 📄 Licence

MIT License — voir [LICENSE](LICENSE) pour les détails.
