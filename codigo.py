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
    0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon', 2: 'bart_simpson',
    3: 'charles_montgomery_burns', 4: 'chief_wiggum', 5: 'comic_book_guy',
    6: 'edna_krabappel', 7: 'homer_simpson', 8: 'kent_brockman',
    9: 'krusty_the_clown', 10: 'lisa_simpson', 11: 'marge_simpson',
    12: 'milhouse_van_houten', 13: 'moe_szyslak', 14: 'ned_flanders',
    15: 'nelson_muntz', 16: 'principal_skinner', 17: 'sideshow_bob'
}
MAP_INDEX_TO_CHAR = {i: name for i, name in MAP_CHARACTERS.items()}

# --- Carga del modelo y clasificadores ---
model = load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Procesamiento de video ---
cap = cv2.VideoCapture(VIDEO_PATH)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Expandir la región para intentar cubrir más del cuerpo
        padding = int(h * 0.6)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)

        if x2 > x1 and y2 > y1:
            personaje_roi = frame[y1:y2, x1:x2]
            if personaje_roi.size > 0:
                try:
                    resized = cv2.resize(personaje_roi, (IMG_SIZE, IMG_SIZE))
                    input_img = np.expand_dims(resized, axis=0) / 255.0
                    prediction = model.predict(input_img, verbose=0)
                    idx = np.argmax(prediction)
                    character = MAP_INDEX_TO_CHAR.get(idx, "Desconocido")

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 255, 0), 2)
                except Exception as e:
                    print("Error al procesar ROI válida:", e)

    cv2.imshow('Video - Deteccion de personajes', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()