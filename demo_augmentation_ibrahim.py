# Contribution de : Ait kadiss Ibrahim
# Démonstration de l'augmentation de données (Data Augmentation).
# Cette technique est cruciale pour éviter l'overfitting dans les modèles CNN et DenseNet.

import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def demo_augmentation():
    print("Génération d'images augmentées en cours...")
    # 1. Charger une image du dataset (la première image est une bottine)
    (x_train, _), _ = fashion_mnist.load_data()
    image_originale = x_train[0]
    
    # Keras s'attend à un format (batch, hauteur, largeur, canaux)
    image_formatee = image_originale.reshape((1, 28, 28, 1))
    
    # 2. Configurer le générateur d'images avec plusieurs transformations
    datagen = ImageDataGenerator(
        rotation_range=30,      # Rotation aléatoire jusqu'à 30 degrés
        width_shift_range=0.2,  # Décalage horizontal
        height_shift_range=0.2, # Décalage vertical
        horizontal_flip=True,   # Retournement horizontal (effet miroir)
        zoom_range=0.2          # Zoom aléatoire
    )
    
    # 3. Préparer l'affichage
    plt.figure(figsize=(12, 6))
    plt.suptitle("Technique de Data Augmentation pour CNN/DenseNet", fontsize=16)
    
    # Afficher l'image originale
    plt.subplot(2, 4, 1)
    plt.imshow(image_originale, cmap='gray')
    plt.title("Image Originale")
    plt.axis('off')
    
    # Générer et afficher 7 variantes de cette même image
    iterateur = datagen.flow(image_formatee, batch_size=1)
    for i in range(2, 9):
        plt.subplot(2, 4, i)
        batch = iterateur.next()
        image_generee = batch[0].astype('uint8').reshape(28, 28)
        plt.imshow(image_generee, cmap='gray')
        plt.title(f"Variante {i-1}")
        plt.axis('off')
        
    plt.tight_layout()
    nom_fichier = 'demo_augmentation.png'
    plt.savefig(nom_fichier)
    print(f"Succès ! L'image illustrant les transformations a été sauvegardée sous '{nom_fichier}'.")

if __name__ == "__main__":
    demo_augmentation()