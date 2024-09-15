import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def detect_sos(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    pinky_tip = hand_landmarks.landmark[20]
    thumb_pinky_distance = abs(thumb_tip.y - pinky_tip.y)
    return thumb_pinky_distance < 0.1


def detect_thumbs_up(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    index_mcp = hand_landmarks.landmark[5]
    return thumb_tip.y < index_mcp.y


cap = cv2.VideoCapture(0)


sos_count = 0
thumbs_up_count = 0
required_frames = 5  


with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        
        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)

        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                
                if detect_sos(hand_landmarks):
                    sos_count += 1
                    thumbs_up_count = 0
                elif detect_thumbs_up(hand_landmarks):
                    thumbs_up_count += 1
                    sos_count = 0
                else:
                    sos_count = 0
                    thumbs_up_count = 0

                
                if sos_count >= required_frames:
                    cv2.putText(image, "SOS Detected! Need Help!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                elif thumbs_up_count >= required_frames:
                    cv2.putText(image, "Thumbs Up, I'm Safe!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        
        cv2.imshow('Gesture Detection', image)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
