import cv2

# Load Haar Cascades — all built into OpenCV, nothing to install
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_smile.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 1: Detect faces
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=3, minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        face_gray = gray[y:y + h, x:x + w]
        face_color = frame[y:y + h, x:x + w]

        # Step 2: Detect smile within the face region
        smiles = smile_cascade.detectMultiScale(
            face_gray, scaleFactor=1.05, minNeighbors=3, minSize=(25, 25)
        )

        # Step 3: Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(
            face_gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)
        )

        # Step 4: Determine emotion from what's detected
        if len(smiles) > 0:
            emotion = "Happy :)"
            color = (0, 255, 0)      # Green
        elif len(eyes) == 0:
            emotion = "Eyes Closed"
            color = (255, 255, 0)    # Yellow
        else:
            emotion = "Neutral :|"
            color = (0, 165, 255)    # Orange

        # Draw face box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Display emotion label
        cv2.putText(
            frame, emotion, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )

        # Draw eye boxes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 1)

    # Face count display
    cv2.putText(
        frame, f'Faces: {len(faces)}', (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )

    cv2.imshow('Real-Time Face & Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()