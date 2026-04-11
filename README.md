# 🧠 Guide Complet : Transfer Learning sur FashionMNIST
## Du zéro absolu à la maîtrise — pour Google Colab Pro

---

## TABLE DES MATIÈRES

1. [C'est quoi le contexte global ?](#1-contexte-global)
2. [Les prérequis à comprendre avant tout](#2-prérequis)
3. [Le notebook qu'on t'a donné — décrypté](#3-le-notebook-décrypté)
4. [Le vrai travail demandé : Transfer Learning](#4-le-travail-demandé)
5. [Comment le faire, étape par étape](#5-comment-le-faire)
6. [Le code complet prêt pour Colab](#6-code-complet)
7. [Comment tout comprendre — la Big Picture](#7-big-picture)

---

## 1. CONTEXTE GLOBAL

### De quoi on parle ?

Imagine que tu veux apprendre à un ordinateur à **reconnaître des vêtements** dans des photos. C'est exactement ça le sujet.

**FashionMNIST** est un jeu de données (dataset) très connu qui contient **70 000 images** de vêtements en niveaux de gris (noir et blanc), réparties en **10 catégories** :

| Numéro | Catégorie    |
|--------|-------------|
| 0      | T-shirt/top |
| 1      | Pantalon    |
| 2      | Pull        |
| 3      | Robe        |
| 4      | Manteau     |
| 5      | Sandale     |
| 6      | Chemise     |
| 7      | Basket      |
| 8      | Sac         |
| 9      | Bottine     |

Chaque image fait **28×28 pixels**, en **1 seul canal** (gris). C'est minuscule.

### Qu'est-ce qu'on essaie de faire ?

On veut construire un programme (un **modèle**) qui, quand on lui montre une de ces images, dit correctement : "c'est un pantalon", "c'est un sac", etc.

### Pourquoi c'est intéressant ?

Parce qu'on va utiliser **deux approches** :

1. **Approche 1 (déjà faite dans le notebook)** : Construire un réseau de neurones **from scratch** (à partir de zéro).
2. **Approche 2 (ce qu'on te demande de faire)** : Utiliser un modèle **déjà entraîné** sur des millions d'images (ImageNet) et l'**adapter** pour notre problème. C'est le **Transfer Learning**.

---

## 2. PRÉREQUIS — Les concepts à connaître

### 2.1 Qu'est-ce qu'un réseau de neurones (Neural Network) ?

Pense à un filtre intelligent. Tu mets une image en entrée, le réseau fait des calculs internes (à travers des "couches"), et à la sortie il donne une **prédiction** (ex: "c'est une basket, confiance 92%").

```
IMAGE → [Couche 1] → [Couche 2] → ... → [Couche N] → PRÉDICTION
```

Chaque couche contient des **poids** (des nombres). Pendant l'**entraînement**, ces poids sont ajustés pour que le réseau fasse de moins en moins d'erreurs.

### 2.2 Qu'est-ce qu'un CNN (Convolutional Neural Network) ?

Un CNN est un type spécial de réseau de neurones conçu spécifiquement pour les **images**. Il a des couches spéciales :

- **Couche de convolution (Conv2d)** : Elle fait glisser un petit filtre (ex: 3×3 pixels) sur toute l'image pour détecter des motifs (bords, textures, formes).
- **Couche de pooling (MaxPool2d)** : Elle réduit la taille de l'image en gardant les infos importantes (comme un zoom arrière intelligent).
- **Couche fully-connected (Linear)** : À la fin, elle prend tous les motifs détectés et décide la classe finale.

**Analogie** : Les premières couches détectent des choses simples (lignes, bords). Les couches du milieu détectent des motifs (textures, formes). Les dernières couches reconnaissent des objets complets (un sac, une chaussure).

### 2.3 Qu'est-ce que l'entraînement ?

C'est un cycle qui se répète des milliers de fois :

1. **Forward pass** : On donne une image au réseau, il prédit une classe.
2. **Calcul de la loss** : On compare la prédiction avec la vraie réponse. La "loss" (perte) mesure à quel point le réseau s'est trompé.
3. **Backward pass** : On calcule comment chaque poids a contribué à l'erreur.
4. **Mise à jour** : On ajuste les poids pour réduire l'erreur.

Un **epoch** = une passe complète sur TOUTES les images d'entraînement.

### 2.4 Qu'est-ce qu'ImageNet ?

ImageNet est un **énorme** dataset : **14 millions d'images**, **1000 catégories** (chiens, voitures, avions, etc.). Des modèles comme ResNet, VGG, AlexNet ont été entraînés dessus pendant des jours/semaines sur des GPU puissants. Ces modèles ont appris à reconnaître des textures, formes, et objets de manière très générale.

### 2.5 Qu'est-ce que le Transfer Learning ?

**L'idée clé** : Pourquoi repartir de zéro quand quelqu'un a déjà fait le gros du travail ?

C'est comme si tu voulais apprendre le portugais en sachant déjà le français et l'espagnol. Tu ne repars pas de zéro — tu **transfères** tes connaissances existantes.

Concrètement :
- On prend un modèle **déjà entraîné** sur ImageNet (il sait déjà détecter des bords, textures, formes).
- On **gèle** (freeze) les premières couches (on garde ce qu'il a appris).
- On **remplace** la dernière couche (celle qui classifie en 1000 classes ImageNet) par une nouvelle couche qui classifie en **10 classes** (nos vêtements).
- On **ré-entraîne** seulement cette dernière couche (ou quelques couches).

### 2.6 Qu'est-ce que "geler des couches" ?

Geler = dire au programme : **"ne touche pas à ces poids, ne les modifie pas pendant l'entraînement"**.

En code, ça se fait avec :
```python
for param in modele.parameters():
    param.requires_grad = False  # gelé = pas de mise à jour
```

`requires_grad = False` signifie : "pas besoin de calculer le gradient (la direction de correction) pour ce poids".

**Pourquoi geler ?**
- Les premières couches ont appris des choses **universelles** (bords, textures) → on les garde.
- Ça accélère l'entraînement (moins de calculs).
- Ça évite le **surapprentissage** (overfitting) quand on a peu de données.

### 2.7 Le problème de compatibilité

Voici LE problème central que tu dois résoudre :

| Propriété | FashionMNIST | Modèles ImageNet |
|-----------|-------------|-----------------|
| Taille image | 28 × 28 | 224 × 224 |
| Canaux | 1 (gris) | 3 (RGB) |
| Normalisation | Aucune spécifique | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |
| Nb classes sortie | 10 | 1000 |

**Solutions** :
1. **Taille** : Redimensionner 28×28 → 224×224 avec `transforms.Resize(224)`.
2. **Canaux** : Dupliquer le canal gris 3 fois : gris → (gris, gris, gris) = pseudo-RGB.
3. **Normalisation** : Appliquer la normalisation ImageNet.
4. **Dernière couche** : Remplacer la couche de sortie (1000 classes → 10 classes).

---

## 3. LE NOTEBOOK QU'ON T'A DONNÉ — DÉCRYPTÉ

Le notebook fait les choses suivantes, dans l'ordre :

### Partie A — CNN from scratch (Cellules 0-34)
- **Cellules 0-6** : Chargement de FashionMNIST, visualisation des images.
- **Cellules 8-12** : Définition d'un petit CNN maison (2 blocs de convolution + couches linéaires).
- **Cellules 13-18** : Entraînement du CNN et visualisation de la loss.
- **Cellules 19-22** : Évaluation de la performance (accuracy, train/test).
- **Cellules 23-34** : Prédictions, matrice de confusion, classification report.

### Partie B — Début du Transfer Learning (Cellules 45-58)
- **Cellule 46** : Transformation des images FashionMNIST (gris → RGB, normalisation ImageNet).
- **Cellule 49** : Chargement de modèles pré-entraînés (AlexNet, ResNet18, VGG16).
- **Cellule 50** : Modification de la dernière couche (1000 → 10 classes).
- **Cellules 52-56** : Gel des couches d'AlexNet + entraînement.

**Ce qui manque / ce qu'on te demande** : Un travail complet et structuré de Transfer Learning avec gel partiel des couches, sur plusieurs modèles.

---

## 4. LE TRAVAIL DEMANDÉ — Résumé clair

On te demande de :

1. **Charger FashionMNIST** avec les bonnes transformations (resize, RGB, normalisation ImageNet).
2. **Charger des modèles pré-entraînés** (ResNet18, VGG16, EfficientNet, etc.).
3. **Geler tout ou partie** des couches de ces modèles.
4. **Remplacer la dernière couche** pour 10 classes.
5. **Entraîner** seulement les couches non-gelées.
6. **Évaluer** les performances.
7. **Comparer** les différentes stratégies de gel.

---

## 5. COMMENT LE FAIRE — Étape par étape

### Étape 1 : Préparer les données

```python
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Transformations pour adapter FashionMNIST aux modèles ImageNet
transform = transforms.Compose([
    transforms.Resize(224),                          # 28×28 → 224×224
    transforms.Grayscale(num_output_channels=3),     # 1 canal → 3 canaux
    transforms.ToTensor(),                           # Image → Tensor [0,1]
    transforms.Normalize(                            # Normalisation ImageNet
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_data = datasets.FashionMNIST(root='data', train=True,  download=True, transform=transform)
test_data  = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)
```

**Explication ligne par ligne** :
- `Resize(224)` : Les modèles ImageNet s'attendent à du 224×224. On agrandit nos petites images.
- `Grayscale(num_output_channels=3)` : On duplique le canal gris pour simuler du RGB.
- `ToTensor()` : Convertit l'image PIL en tenseur PyTorch (les nombres deviennent des nombres décimaux entre 0 et 1).
- `Normalize(mean, std)` : Applique exactement la même normalisation qui a été utilisée pendant l'entraînement sur ImageNet. Sans ça, le modèle est "perdu".

### Étape 2 : Charger un modèle pré-entraîné

```python
from torchvision import models
import torch.nn as nn

# Charger ResNet18 avec les poids pré-entraînés sur ImageNet
model = models.resnet18(weights='IMAGENET1K_V1')
```

Le modèle téléchargé a été entraîné pendant des jours sur ImageNet. Ses poids encodent toutes les connaissances apprises.

### Étape 3 : Geler les couches

**Stratégie A — Tout geler sauf la dernière couche :**
```python
# Geler TOUTES les couches
for param in model.parameters():
    param.requires_grad = False

# Remplacer la dernière couche (elle sera dégelée par défaut)
model.fc = nn.Linear(model.fc.in_features, 10)
# model.fc.in_features = 512 pour ResNet18
# 10 = nos 10 classes FashionMNIST
```

**Stratégie B — Geler seulement les premières couches :**
```python
# D'abord tout geler
for param in model.parameters():
    param.requires_grad = False

# Dégeler les 2 derniers blocs (layer3, layer4) + fc
for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Linear(model.fc.in_features, 10)
```

**Stratégie C — Ne rien geler (fine-tuning complet) :**
```python
# On garde tous les requires_grad = True (par défaut)
model.fc = nn.Linear(model.fc.in_features, 10)
```

### Étape 4 : Configurer l'entraînement

```python
import torch
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()

# IMPORTANT : on optimise seulement les paramètres non-gelés
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001
)
```

**Pourquoi `filter(lambda p: p.requires_grad, ...)`** : On dit à l'optimiseur de ne mettre à jour QUE les poids qui ne sont pas gelés. Ça économise de la mémoire et du temps.

### Étape 5 : Entraîner

```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # Remettre les gradients à 0
        outputs = model(images)        # Forward pass
        loss = criterion(outputs, labels)  # Calcul de la perte
        loss.backward()                # Backward pass (calcul des gradients)
        optimizer.step()               # Mise à jour des poids

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), 100. * correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():  # Pas de calcul de gradient en évaluation
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), 100. * correct / total
```

```python
# Boucle d'entraînement
n_epochs = 5
for epoch in range(n_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc   = evaluate(model, test_loader, criterion, device)
    print(f"Epoch {epoch+1}/{n_epochs} | "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.1f}% | "
          f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.1f}%")
```

### Étape 6 : Adapter pour d'autres modèles

Chaque modèle a une architecture différente. Voici comment remplacer la dernière couche :

| Modèle | Dernière couche | Code de remplacement |
|--------|----------------|---------------------|
| ResNet18 | `model.fc` | `model.fc = nn.Linear(512, 10)` |
| VGG16 | `model.classifier[6]` | `model.classifier[6] = nn.Linear(4096, 10)` |
| AlexNet | `model.classifier[6]` | `model.classifier[6] = nn.Linear(4096, 10)` |
| EfficientNet-B0 | `model.classifier[1]` | `model.classifier[1] = nn.Linear(1280, 10)` |
| DenseNet121 | `model.classifier` | `model.classifier = nn.Linear(1024, 10)` |

**Astuce universelle** : Pour trouver `in_features` automatiquement :
```python
# Pour ResNet :
model.fc = nn.Linear(model.fc.in_features, 10)
# Pour VGG/AlexNet :
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)
```

---

## 6. CODE COMPLET PRÊT POUR COLAB

Copie-colle ce code dans un notebook Colab (GPU activé : Runtime → Change runtime type → T4 GPU).

```python
#############################################
# CELLULE 1 : Imports et configuration
#############################################
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Utilisation de : {device}")

classes = ['T-shirt', 'Pantalon', 'Pull', 'Robe', 'Manteau',
           'Sandale', 'Chemise', 'Basket', 'Sac', 'Bottine']

#############################################
# CELLULE 2 : Préparation des données
#############################################
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_data = datasets.FashionMNIST('data', train=True,  download=True, transform=transform)
test_data  = datasets.FashionMNIST('data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True,  num_workers=2)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False, num_workers=2)

print(f"Train: {len(train_data)} images | Test: {len(test_data)} images")

#############################################
# CELLULE 3 : Fonctions d'entraînement
#############################################
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / len(loader), 100. * correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return running_loss / len(loader), 100. * correct / total

def run_experiment(model, name, n_epochs=5, lr=0.001):
    """Lance un entraînement complet et retourne les résultats."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

    print(f"\n{'='*50}")
    print(f"  Expérience : {name}")
    nb_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    nb_total     = sum(p.numel() for p in model.parameters())
    print(f"  Paramètres entraînables : {nb_trainable:,} / {nb_total:,}")
    print(f"{'='*50}")

    for epoch in range(n_epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        history['train_loss'].append(tr_loss)
        history['test_loss'].append(te_loss)
        history['train_acc'].append(tr_acc)
        history['test_acc'].append(te_acc)
        print(f"  Epoch {epoch+1}/{n_epochs} | "
              f"Train: {tr_acc:.1f}% | Test: {te_acc:.1f}%")

    return history

#############################################
# CELLULE 4 : Expérience 1 — ResNet18 tout gelé sauf fc
#############################################
resnet = models.resnet18(weights='IMAGENET1K_V1')
for param in resnet.parameters():
    param.requires_grad = False
resnet.fc = nn.Linear(resnet.fc.in_features, 10)

hist_resnet_frozen = run_experiment(resnet, "ResNet18 — Tout gelé sauf fc", n_epochs=5)

#############################################
# CELLULE 5 : Expérience 2 — ResNet18 gel partiel
#############################################
resnet2 = models.resnet18(weights='IMAGENET1K_V1')
for param in resnet2.parameters():
    param.requires_grad = False
# Dégeler layer4 + fc
for param in resnet2.layer4.parameters():
    param.requires_grad = True
resnet2.fc = nn.Linear(resnet2.fc.in_features, 10)

hist_resnet_partial = run_experiment(resnet2, "ResNet18 — Gel partiel (layer4 dégelé)", n_epochs=5)

#############################################
# CELLULE 6 : Expérience 3 — VGG16 tout gelé sauf classifier
#############################################
vgg = models.vgg16(weights='IMAGENET1K_V1')
for param in vgg.features.parameters():
    param.requires_grad = False
vgg.classifier[6] = nn.Linear(4096, 10)

hist_vgg = run_experiment(vgg, "VGG16 — Features gelées", n_epochs=5)

#############################################
# CELLULE 7 : Comparaison des résultats
#############################################
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for hist, label in [(hist_resnet_frozen, 'ResNet18 tout gelé'),
                    (hist_resnet_partial, 'ResNet18 gel partiel'),
                    (hist_vgg, 'VGG16 gelé')]:
    axes[0].plot(hist['test_acc'], label=label)
    axes[1].plot(hist['test_loss'], label=label)

axes[0].set_title('Test Accuracy par Epoch')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy (%)')
axes[0].legend(); axes[0].grid(True)

axes[1].set_title('Test Loss par Epoch')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
axes[1].legend(); axes[1].grid(True)

plt.tight_layout()
plt.show()
```

---

## 7. LA BIG PICTURE — Comment tout s'emboîte

### Le schéma mental à retenir

```
┌─────────────────────────────────────────────────────────┐
│           MODÈLE PRÉ-ENTRAÎNÉ (ex: ResNet18)           │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │
│  │ Couches  │→ │ Couches  │→ │ Couches  │→ │Dernière│  │
│  │ basses   │  │ moyennes │  │ hautes   │  │couche  │  │
│  │          │  │          │  │          │  │(fc)    │  │
│  │ Détecte: │  │ Détecte: │  │ Détecte: │  │        │  │
│  │ bords,   │  │ textures,│  │ objets,  │  │ 1000   │  │
│  │ lignes   │  │ motifs   │  │ formes   │  │ classes│  │
│  └──────────┘  └──────────┘  └──────────┘  └────────┘  │
│   🔒 GELÉ      🔒 GELÉ       🔓 ou 🔒     🔄 REMPLACÉ │
│                                             par 10 cls  │
└─────────────────────────────────────────────────────────┘
```

### Les 3 stratégies de gel, résumées

| Stratégie | Quoi geler | Quand l'utiliser | Vitesse | Risque d'overfitting |
|-----------|-----------|-----------------|---------|---------------------|
| **Tout geler** sauf la dernière couche | Toutes les couches sauf `fc` | Peu de données, ou données similaires à ImageNet | ⚡ Très rapide | Très faible |
| **Gel partiel** | Couches basses gelées, couches hautes libres | Données modérément différentes d'ImageNet | 🏃 Moyen | Moyen |
| **Rien geler** (fine-tuning total) | Rien | Beaucoup de données, très différentes d'ImageNet | 🐢 Lent | Élevé |

### Pourquoi ça marche ?

Les couches basses d'un CNN apprennent des **features universelles** (bords, gradients de couleur, textures simples). Ces features sont utiles pour TOUTE tâche de vision. Un bord est un bord, qu'il s'agisse d'une photo de chat ou d'un t-shirt.

En gelant ces couches, on réutilise ce savoir universel et on n'entraîne que les couches qui doivent se spécialiser pour notre tâche spécifique (distinguer un pull d'une chemise).

### Flux complet en un coup d'œil

```
1. Image FashionMNIST (28×28, gris)
         │
         ▼
2. Transformations : Resize(224) + Grayscale→RGB + Normalize
         │
         ▼
3. Image transformée (224×224, 3 canaux, normalisée)
         │
         ▼
4. Modèle pré-entraîné (couches gelées = 🔒)
         │
         ▼
5. Nouvelle dernière couche (10 sorties)
         │
         ▼
6. Prédiction : "C'est un sac !" (classe 8)
         │
         ▼
7. Comparaison avec la vraie réponse → Calcul de la loss
         │
         ▼
8. Mise à jour UNIQUEMENT des couches non-gelées
         │
         ▼
9. Répéter pour chaque batch × chaque epoch
```

### Glossaire rapide

| Terme | Signification |
|-------|--------------|
| **Epoch** | Une passe complète sur toutes les données d'entraînement |
| **Batch** | Un petit groupe d'images traitées ensemble (ex: 64) |
| **Loss** | Mesure de l'erreur du modèle (plus c'est bas, mieux c'est) |
| **Accuracy** | Pourcentage de bonnes prédictions |
| **Gradient** | Direction dans laquelle ajuster les poids pour réduire la loss |
| **requires_grad** | Si True, PyTorch calcule le gradient pour ce poids |
| **model.train()** | Active le mode entraînement (dropout actif, etc.) |
| **model.eval()** | Active le mode évaluation (dropout désactivé) |
| **torch.no_grad()** | Désactive le calcul des gradients (économise mémoire) |
| **optimizer.zero_grad()** | Remet les gradients à zéro avant chaque batch |

---

### Conseils pour Colab Pro

- Active le GPU : **Runtime → Change runtime type → T4 GPU**.
- Avec le batch_size=64 et Resize(224), le VGG16 consomme beaucoup de RAM GPU. Si tu as des erreurs "CUDA out of memory", réduis le batch_size à 32 ou 16.
- L'entraînement de 5 epochs avec ResNet18 gelé prend environ 10-15 min sur T4.

---

*Ce guide t'a emmené du point zéro jusqu'à la compréhension complète du Transfer Learning appliqué à FashionMNIST. Tu sais maintenant ce qu'est un CNN, pourquoi on réutilise des modèles pré-entraînés, comment geler des couches, et comment adapter n'importe quel modèle ImageNet pour ton propre problème.*
