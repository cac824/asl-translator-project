import mediapipe as mp
import cv2
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions
from mediapipe.tasks.python import BaseOptions
from collections import deque

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load Gesture Recognizer model
model_path = "C:/Users/panda/PycharmProjects/pythonProblems/asl translator/gesture_recognizer.task"
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=mp.tasks.vision.RunningMode.VIDEO
)
trajectory = deque(maxlen=15)
cap = cv2.VideoCapture(0)
with GestureRecognizer.create_from_options(options) as recognizer, mp_hands.Hands(min_detection_confidence=0.5,
                                                                                  min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame.")
            break

        # turn frame to rgb
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # processes the rgb frame
        results = hands.process(rgb_frame)
        # hand detection
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # hand landmarks and threshold
                thresh = 0.18
                # wrist landmark
                wrist_y = hand_landmarks.landmark[0].y
                wrist_x = hand_landmarks.landmark[0].x
                # thumb landmark
                thumb_cmc_x = hand_landmarks.landmark[1].x
                thumb_cmc_y = hand_landmarks.landmark[1].y
                thumb_mcp_x = hand_landmarks.landmark[2].x
                thumb_mcp_y = hand_landmarks.landmark[2].y
                thumb_ip_x = hand_landmarks.landmark[3].x
                thumb_ip_y = hand_landmarks.landmark[3].y
                thumb_tip_x = hand_landmarks.landmark[4].x
                thumb_tip_y = hand_landmarks.landmark[4].y
                # index finger landmark
                index_mcp_y = hand_landmarks.landmark[5].y
                index_mcp_x = hand_landmarks.landmark[5].x
                index_pip_y = hand_landmarks.landmark[6].y
                index_pip_x = hand_landmarks.landmark[6].x
                index_dip_y = hand_landmarks.landmark[7].y
                index_dip_x = hand_landmarks.landmark[7].x
                index_tip_x = hand_landmarks.landmark[8].x
                index_tip_y = hand_landmarks.landmark[8].y
                index_tip_z = hand_landmarks.landmark[8].z
                # middle finger landmark
                middle_mcp_y = hand_landmarks.landmark[9].y
                middle_mcp_x = hand_landmarks.landmark[9].x
                middle_pip_y = hand_landmarks.landmark[10].y
                middle_pip_x = hand_landmarks.landmark[10].x
                middle_dip_y = hand_landmarks.landmark[11].y
                middle_dip_x = hand_landmarks.landmark[11].x
                middle_tip_x = hand_landmarks.landmark[12].x
                middle_tip_y = hand_landmarks.landmark[12].y
                middle_tip_z = hand_landmarks.landmark[12].z
                # ring finger landmark
                ring_mcp_y = hand_landmarks.landmark[13].y
                ring_mcp_x = hand_landmarks.landmark[13].x
                ring_pip_y = hand_landmarks.landmark[14].y
                ring_pip_x = hand_landmarks.landmark[14].x
                ring_dip_y = hand_landmarks.landmark[15].y
                ring_dip_x = hand_landmarks.landmark[15].x
                ring_tip_x = hand_landmarks.landmark[16].x
                ring_tip_y = hand_landmarks.landmark[16].y
                # pinky finger landmark
                pinky_mcp_y = hand_landmarks.landmark[17].y
                pinky_mcp_x = hand_landmarks.landmark[17].x
                pinky_pip_y = hand_landmarks.landmark[18].y
                pinky_pip_x = hand_landmarks.landmark[18].x
                pinky_dip_y = hand_landmarks.landmark[19].y
                pinky_dip_x = hand_landmarks.landmark[19].x
                pinky_tip_x = hand_landmarks.landmark[20].x
                pinky_tip_y = hand_landmarks.landmark[20].y
                print("thumb {}".format(index_tip_z))
                print("index {}".format(index_tip_x))
                print("middle {}".format(middle_tip_z))
                print("ring {}".format(middle_tip_x))
                print("pinky {}".format(ring_pip_y))

                # finger extension
                index_extended = index_tip_y < index_pip_y < index_mcp_y
                middle_extended = middle_tip_y < middle_pip_y < middle_mcp_y
                ring_extended = ring_tip_y < ring_pip_y < ring_mcp_y
                pinky_extended = pinky_tip_y < pinky_pip_y < pinky_mcp_y
                # finger orientation
                thumb_sideways = abs(thumb_tip_x - thumb_mcp_x) > 0.09
                thumb_vertical = abs(thumb_tip_y - thumb_mcp_y) < 0.5
                index_sideways = abs(index_tip_x - index_mcp_x) > 0.09
                thumb_down = thumb_tip_y > thumb_ip_y
                index_vertical = abs(index_tip_y - index_mcp_y) > 0.09
                index_down = index_tip_y > index_mcp_y
                middle_sideways = abs(middle_tip_x - middle_mcp_x) > 0.2
                middle_vertical = abs(middle_tip_y - middle_mcp_y) > 0.09

                # finger folded
                index_folded = index_tip_y > index_pip_y < wrist_y or index_tip_y < index_pip_y > wrist_y
                middle_folded = middle_tip_y > middle_pip_y < wrist_y or middle_tip_y < middle_pip_y > wrist_y
                ring_folded = ring_tip_y > ring_pip_y < wrist_y or ring_tip_y < ring_pip_y > wrist_y
                pinky_folded = pinky_tip_y > pinky_pip_y < wrist_y or pinky_tip_y < pinky_pip_y > wrist_y
                thumb_folded = thumb_tip_x > thumb_mcp_x < wrist_y or thumb_tip_y < thumb_mcp_y > wrist_y

                # appends x and y coords to the trajectory
                trajectory.append((pinky_tip_y, pinky_tip_x))


                def mid_point(point1x, point1y, point2x, point2y):  # calculates midpoint
                    return ((point1x - point2x) ** 2 + (point1y - point2y) ** 2) ** 0.5


                letter = None  # sets the variable to nothing for the sign detection to assign the variable

                # sign detection
                if not (pinky_pip_y and ring_pip_y > index_pip_y and middle_pip_y) == False:
                    if thumb_tip_y < index_pip_y and thumb_tip_y < middle_pip_y and thumb_tip_y < ring_pip_y and thumb_tip_y < pinky_pip_y:
                        letter = "A"
                if thumb_tip_y > index_pip_y and thumb_tip_y > middle_pip_y and thumb_tip_y > ring_pip_y and thumb_tip_y > pinky_pip_y:
                    letter = "B"
                if index_extended and middle_extended and ring_extended and pinky_extended:
                    if thumb_tip_y - index_tip_y > thresh and thumb_tip_y > index_tip_y:
                        letter = "C"
                if index_tip_y < thumb_tip_y and index_tip_y < middle_tip_y and index_tip_y < ring_tip_y and index_tip_y < pinky_tip_y:
                    if middle_folded and ring_folded and pinky_folded:
                        letter = "D"
                if index_tip_y > index_pip_y and middle_tip_y > middle_pip_y and ring_tip_y > ring_pip_y and pinky_tip_y > pinky_pip_y:
                    if thumb_tip_y > index_tip_y:
                        letter = "E"
                if middle_extended and ring_extended and pinky_extended:
                    if index_folded and abs(index_tip_y - thumb_tip_y) <= 0.15:
                        letter = "F"
                if index_sideways and middle_folded and ring_folded and pinky_folded and abs(
                        index_tip_y - thumb_tip_y) <= 0.15:
                    letter = "G"
                if index_extended and middle_extended and ring_folded and pinky_folded:
                    if middle_sideways and index_sideways:
                        letter = "H"
                if pinky_extended and index_folded and middle_folded and ring_folded:
                    letter = "I"
                if len(trajectory) == trajectory.maxlen:
                    start_x, start_y = trajectory[0]  # first point of motion
                    mid_x, mid_y = trajectory[len(trajectory) // 2]  # when the J curves
                    end_x, end_y = trajectory[-1]  # the end of the motion

                    moved_down = abs(end_y - start_y) > 0.1  # check downward movement
                    curved = abs(end_x - start_x) > 0.06  # check lateral curve

                    if moved_down and curved and index_folded and middle_folded and ring_folded:
                        if pinky_extended:
                            letter = "J"
                if index_extended and middle_extended and ring_folded and pinky_folded:
                    if index_pip_y < thumb_tip_y < index_mcp_y and middle_pip_y < thumb_tip_y < middle_mcp_y:
                        letter = "K"
                if index_extended and thumb_sideways and middle_folded and ring_folded and pinky_folded:
                    letter = "L"
                if pinky_pip_y > index_pip_y and pinky_pip_y > middle_pip_y and pinky_pip_y > ring_pip_y:
                    if abs(thumb_tip_x - index_pip_y) >= 0.1 and abs(thumb_tip_x - middle_pip_y) >= 0.1 and abs(
                            thumb_tip_x - ring_pip_y) >= 0.1:
                        if index_folded and middle_folded and ring_folded and pinky_folded:
                            letter = "M"
                if pinky_pip_y and ring_pip_y > index_pip_y and middle_pip_y:
                    if abs(thumb_tip_x - index_pip_y) >= 0.1 and abs(thumb_tip_x - middle_pip_y) >= 0.1 and abs(
                            thumb_tip_x - ring_pip_y) >= 0.1:
                        if index_folded and middle_folded and ring_folded and pinky_folded and not index_sideways:
                            letter = "N"
                if int(index_tip_y * 10) and int(middle_tip_x * 10) and int(ring_tip_x * 10) and int(
                        pinky_tip_x * 10) == 5:
                    if index_folded and middle_folded and ring_folded and pinky_folded:
                        letter = "O"
                if index_sideways and ring_folded and pinky_folded and middle_vertical:
                    if middle_tip_y > ring_pip_y and pinky_pip_y:
                        if thumb_tip_x < index_dip_x and middle_dip_x:
                            letter = "P"
                if index_tip_y > wrist_y and thumb_tip_y > wrist_y:
                    if middle_folded and ring_folded and pinky_folded:
                        letter = "Q"  # S T U V W X Y Z
                if abs(index_tip_z - middle_tip_z) <= 0.02 and index_extended and middle_extended and ring_folded and pinky_folded:
                    letter = "R"

                if letter:  # rewrites the label each time a new letter gets assigned
                    cv2.rectangle(frame, (45, 20), (200, 60), (0, 0, 0), -1)  #
                    cv2.putText(frame, letter, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2)
                mp_drawing.draw_landmarks(frame, hand_landmarks,
                                          mp_hands.HAND_CONNECTIONS)  # draws the connections on the hands

        cv2.imshow('Hand Detection', frame)

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
