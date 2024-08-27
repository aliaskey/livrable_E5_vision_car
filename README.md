Flask Vision car

Cette application Flask permet de réaliser la segmentation d'images à l'aide d'un modèle de deep learning. Elle inclut un tableau de bord de monitoring via Flask Monitoring Dashboard.

Fonctionnalités

- Segmentation d'images : Chargez une image pour obtenir un masque segmenté.
- Monitoring intégré : Surveillez les performances et les accès grâce à Flask Monitoring Dashboard.
- Modèle de segmentation : Utilise un modèle de deep learning basé sur UNet.

Prérequis

- Python 3.7+
- Flask
- TensorFlow
- Keras
- Pillow
- Flask Monitoring Dashboard
- NumPy

Installation

1. Clonez ce dépôt :

    ```bash
    git clone https://github.com/aliaskey/livrable_E5_vision_car.git
    cd votre-repo
    ```

2. Créez et activez un environnement virtuel :

    ```bash
    python -m venv venv
    Sur Windows: venv\Scripts\activate
    ```

3. Installez les dépendances :

    ```bash
    pip install -r requirements.txt
    ```

4. Assurez-vous que le modèle `unet_vgg16_categorical_crossentropy_raw_data.keras` est placé dans le dossier `models/`.

Utilisation

1. Lancer l'application :

    ```bash
    python app.py
    ```

2. Accéder à l'application : Ouvrez votre navigateur et allez à [http://127.0.0.1:8000](http://127.0.0.1:8000).

3. Upload d'image : Utilisez l'interface pour charger une image et obtenir le masque de segmentation.

4. **Tableau de bord** : Accédez à Flask Monitoring Dashboard à l'adresse [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard) pour voir les statistiques.

## Fichiers Importants

- `app.py` : Le script principal de l'application Flask.
- `templates/index.html` : Le fichier HTML pour l'interface utilisateur.
- `models/` : Dossier contenant les modèles de deep learning.

## Configuration

### Compte Administrateur pour le Monitoring Dashboard

Lors du premier lancement, un compte administrateur est automatiquement créé avec les identifiants suivants :

- **Username** : `admin`
- **Password** : `password123`

Vous pouvez modifier ces informations directement dans le script avant de lancer l'application.

### Journalisation

L'application logue les événements dans un fichier `app.log`. Les logs incluent :

- Informations générales sur l'application.
- Erreurs liées au chargement des modèles et au traitement des images.
- Temps de traitement des requêtes.

## Déploiement

Pour déployer cette application sur un serveur de production, il est recommandé d'utiliser un serveur WSGI tel que Gunicorn, ainsi qu'un serveur web comme Nginx pour servir les fichiers statiques et gérer le routage.

