import cv2
import mediapipe as mp
import time
#only works for mediapipe 10.90
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands()
pose = mp_pose.Pose()

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
prev_time = 0
curr_time = 0

prev_wrist_y = 0
last_strum_time = 0
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
            landmarks = pose_result.pose_landmarks.landmark
            right_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

            wrist_y = round(right_wrist.y, 3)

            speed = wrist_y - prev_wrist_y
            speed = round(speed, 3)
            now = time.time()

            if speed > 0.05 and (now - last_strum_time) > 0.4:
                last_strum_time = now
                print("DOWNSTRUM!")
                cv2.putText(frame, "DOWNSTRUM!", (500,80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            elif speed < -0.05 and (now - last_strum_time) > 0.4:
                last_strum_time = now
                print("UPSTRUM!")
                cv2.putText(frame, "UPSTRUM!", (500,80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 165, 0), 3)
            prev_wrist_y = wrist_y
            print("Right Wrist Y:", wrist_y, "| Speed:", speed)

            cv2.putText(frame, "Wrist Y:" + str(wrist_y), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)   
            cv2.putText(frame, "Speed:"  + str(speed), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time
    cv2.putText(frame, "FPS: " + str(int(fps)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("InstrumentalCV", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

