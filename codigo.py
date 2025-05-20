import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# --- Configuración ---
IMG_SIZE = 64
VIDEO_PATH = 'video_simpsons.mp4'
MODEL_PATH = "model.keras"

# --- Mapeo de personajes ---
MAP_CHARACTERS = {
    0: 'abraham_abu', 1: 'apu_n', 2: 'bart_simpson',
    3: 'mr_burns', 4: 'chief_wiggum', 5: 'comic_book_guy',
    6: 'edna_krabappel', 7: 'homer_simpson', 8: 'kent_brockman',
    9: 'krusty', 10: 'lisa_simpson', 11: 'marge_simpson',
    12: 'milhouse', 13: 'moe_szyslak', 14: 'ned_flanders',
    15: 'nelson_muntz', 16: 'principal_skinner', 17: 'sideshow_bob'
}
MAP_INDEX_TO_CHAR = {i: name for i, name in MAP_CHARACTERS.items()}

# --- Carga del modelo ---
model = load_model(MODEL_PATH)

# --- Procesamiento de video ---
cap = cv2.VideoCapture(VIDEO_PATH)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionar el frame para el modelo
    resized_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    input_img = np.expand_dims(resized_frame, axis=0) / 255.0

    # Realizar predicción
    prediction = model.predict(input_img, verbose=0)[0]
    top_indices = prediction.argsort()[-3:][::-1]
    top_predictions = [(MAP_INDEX_TO_CHAR[i], prediction[i]) for i in top_indices]

    # Crear un panel lateral
    height, width, _ = frame.shape
    panel_width = 500
    panel = np.zeros((height, panel_width, 3), dtype=np.uint8)

    y0 = 100
    for i, (name, prob) in enumerate(top_predictions):
        wrapped_name = name.replace('_', ' ').title()
        text = f"{wrapped_name}: {prob:.2f}"
        y = y0 + i * 80
        cv2.putText(panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    combined = np.hstack((frame, panel))
    cv2.imshow('Video - Predicciones', combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()