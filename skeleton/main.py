import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Webcam not found/In use of other application")
        break
    frame = cv2.flip(frame, 1)
    cv2.imshow("InstrumentalCV", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

