import cv2
import mediapipe as mp

#only works for mediapipe 10.90
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands()
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Webcam not found/Is in use of other application")
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_result = hands.process(rgb)
    pose_result = pose.process(rgb)
    if hand_result.multi_hand_landmarks:
        for hand in hand_result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
    if pose_result.pose_landmarks:
            mp_draw.draw_landmarks(frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow("InstrumentalCV", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

