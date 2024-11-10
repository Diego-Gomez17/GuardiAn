import cv2
import torch
from ultralytics import YOLO
import mediapipe as mp
import time

import os

# Crear la carpeta 'captura' si no existe
if not os.path.exists("capturas"):
    os.makedirs("capturas")

# Iniciar captura de video
cap = cv2.VideoCapture(1)

# Inicializar YOLO con el modelo entrenado
model = YOLO("runs/detect/train10/weights/best.pt")

# Inicializar MediaPipe para la detección y seguimiento de manos y rostros
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.1, min_tracking_confidence=0.4)
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=1.0)
mp_drawing = mp.solutions.drawing_utils

cuchillo_detectado = False
riesgo = None
captura_realizada = False

if not cap.isOpened():
    print("Error: No se pudo acceder a la cámara.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el frame de la cámara.")
        break

    # YOLO detección de objetos
    results = model.predict(frame)

    # Dibujar bounding boxes de YOLO
    for detection in results[0].boxes:
        x1, y1, x2, y2 = map(int, detection.xyxy[0])
        score = detection.conf[0]
        class_id = int(detection.cls[0])

        if score >= 0.6 and class_id == 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"CUCHILLO - ARMA BLANCA {model.names[class_id]}: {score:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            if model.names[0]: 
                cuchillo_detectado = True
                riesgo = (x1, y1, x2, y2)

            # Dibujar la caja delimitadora del objeto
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            

    # Convertir el frame a RGB para MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # MediaPipe detección y seguimiento de manos
    results_hands = hands.process(frame_rgb)
    mano_detectada = False

    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
            hand_bbox = []
            for landmark in hand_landmarks.landmark:
                hand_bbox.append((int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])))

            # Verificar si alguno de los puntos de la mano está dentro de la caja del cuchillo
            if cuchillo_detectado and riesgo:
                knife_x1, knife_y1, knife_x2, knife_y2 = riesgo
                for score in hand_bbox:
                    px, py = score
                    # Si el punto de la mano está dentro del bounding box del cuchillo, mostramos el mensaje
                    if knife_x1 < px < knife_x2 and knife_y1 < py < knife_y2:
                        
                        mano_detectada = True
                        cv2.putText(frame, "PELIGRO POTENCIAL", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        cv2.imwrite(f"captura/captura_{timestamp}.png", frame)  # Guardar la imagen con timestamp
                        print(f"Imagen guardada como captura_{timestamp}.png")  # Confirmación en consola

                        break  # Salir del bucle al encontrar el primer punto dentro del cuchillo
            if mano_detectada and cuchillo_detectado:
                cv2.putText(frame, "ASS", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # MediaPipe detección de rostros
    results_face = face_detection.process(frame_rgb)

    if results_face.detections:
        for detection in results_face.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"CARA: {detection.score[0]:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Mostrar el frame con detecciones de YOLO y manos
    cv2.imshow("GuardIAn", frame)

    # Salir con la tecla 'Esc'
    if cv2.waitKey(1) == 27:
        break

# Liberar recursos
cap.release()
hands.close()
face_detection.close()
cv2.destroyAllWindows()
