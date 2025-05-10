import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

# Realted to module of Hand-detection
mpHands = mp.solutions.hands
hands = mpHands.Hands()  # It has the params as static image mode to detect/track hands, max_num_hands, min and max confidence
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark): # Each id has a landmarks this will be used by us for perfoming tasks
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy) #for pixels

                ## Detecting 0 landmark
                if id==0:
                    cv2.circle(img, (cx,cy), 15, (0,255,0), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    if not success:
        print("Ignoring empty frame.")
        continue
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Webcam Feed", img)

    # Break loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break