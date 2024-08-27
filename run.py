
from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.utils import get_custom_objects
import logging
import base64
from io import BytesIO
import flask_monitoringdashboard as dashboard
from flask_monitoringdashboard import bind
from flask_monitoringdashboard.database import session_scope, User
from werkzeug.security import generate_password_hash
import time

app = Flask(__name__)
dashboard.bind(app)

# Configuration de la journalisation
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s'
)

# Création automatique d'un compte administrateur pour Flask Monitoring Dashboard
admin_username = 'admin'
admin_password = 'password123'

with session_scope() as session:
    # Vérifiez si l'utilisateur existe déjà
    user_exists = session.query(User).filter_by(username=admin_username).first()
    if not user_exists:
        # Créer un nouvel utilisateur administrateur avec un mot de passe hashé
        admin_user = User(
            username=admin_username,
            is_admin=True
        )
        admin_user.set_password(admin_password)  # Méthode correcte pour définir le mot de passe, si disponible
        session.add(admin_user)
        session.commit()
        print(f"Admin user created with username '{admin_username}' and password '{admin_password}'")
    else:
        print(f"Admin user with username '{admin_username}' already exists.")

def load_simple_model():
    """Charge un modèle simple pour les tests."""
    try:
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(512, 512, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(),  
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(),  
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(),  
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(),  
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        logging.info("Modèle simple chargé avec succès.")
        return model
    except Exception as e:
        logging.error(f'Erreur lors du chargement du modèle simple: {str(e)}', exc_info=True)
        return None

def load_complex_model():
    """Charge le modèle complexe. En cas d'échec, charge un modèle alternatif ou un modèle simple."""
    try:
        custom_objects = get_custom_objects()
        MODEL_PATH = "models/unet_vgg16_categorical_crossentropy_raw_data.keras"
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
        logging.info("Modèle 'unet_vgg16_categorical_crossentropy_raw_data.keras' chargé avec succès.")
        return model
    except Exception as e:
        logging.error(f'Erreur lors du chargement du modèle principal: {str(e)}', exc_info=True)
        try:
            MODEL_PATH = "models/unet_v2.keras"
            model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
            logging.info("Modèle 'unet_v2.keras' chargé avec succès.")
            return model
        except Exception as e:
            logging.error(f'Erreur lors du chargement du modèle alternatif: {str(e)}', exc_info=True)
            return load_simple_model()

# Charge le modèle à l'initialisation de l'application pour éviter de le recharger à chaque requête
model = load_complex_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    start_time = time.time()  # Commence à chronométrer
    
    logging.info('Upload request received')

    if model is None:
        logging.error('Model not loaded')
        return jsonify({'error': 'Model not loaded'}), 500

    # Définir les couleurs pour la segmentation
    colors = np.array([
        [68, 1, 84],
        [70, 49, 126],
        [54, 91, 140],
        [39, 126, 142],
        [31, 161, 135],
        [73, 193, 109],
        [159, 217, 56],
        [253, 231, 36]
    ])

    if 'file' not in request.files:
        logging.warning('No file part in the request')
        return jsonify({'error': 'No file part in the request'}), 400

    image = request.files['file']
    if image.filename == '':
        logging.warning('No selected file')
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Traitement de l'image
        img = Image.open(image)

        IMAGE_SIZE = 512
        img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.NEAREST)
        img_resized = np.array(img_resized)

        img_resized = np.expand_dims(img_resized, 0)
        img_resized = img_resized / 255.0

        # Prédire le masque
        predict_mask = model.predict(img_resized, verbose=0)
        predict_mask = np.squeeze(predict_mask, axis=0)
        predict_mask = (predict_mask > 0.5).astype(np.uint8)  # Binarisation simple

        # Redimensionner le masque pour qu'il corresponde à l'image d'origine
        predict_mask = Image.fromarray(predict_mask[:, :, 0])  # Utilisation du premier canal (si binaire)
        predict_mask = predict_mask.resize((img.size[0], img.size[1]), resample=Image.Resampling.NEAREST)

        # Appliquer les couleurs au masque prédit
        predict_mask = np.array(predict_mask)
        predict_mask = colors[predict_mask]
        predict_mask = predict_mask.astype(np.uint8)

        # Convertir l'image et le masque en base64 pour l'affichage
        buffered_img = BytesIO()
        img.save(buffered_img, format="PNG")
        base64_img = base64.b64encode(buffered_img.getvalue()).decode("utf-8")

        buffered_mask = BytesIO()
        mask_image = Image.fromarray(predict_mask)
        mask_image.save(buffered_mask, format="PNG")
        base64_mask = base64.b64encode(buffered_mask.getvalue()).decode("utf-8")

        logging.info('Prediction completed successfully')

        processing_time = time.time() - start_time  # Calcule le temps de traitement
        if processing_time > 1.0:  # Seuil de 1 seconde
            logging.warning(f'Processing time exceeded: {processing_time:.2f} seconds')

        return jsonify({'message': "predict ok", "img_data": base64_img, "mask_data": base64_mask})

    except Exception as e:
        logging.error('Error during processing', exc_info=True)
        return jsonify({'error': 'Processing error'}), 500

if __name__ == '__main__':
    logging.info('Starting the Flask app')
    app.run(debug=True, port=8000)