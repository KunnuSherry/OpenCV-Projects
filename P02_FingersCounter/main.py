import cv2
import time
import os
import handTrackingModule as htm
import mediapipe as mp

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
detector = htm.handDetector(detectionCon=0.75)
tipIDS = [4, 8, 12, 16, 20]


#Storing the images
folderPath = os.path.join(os.path.dirname(__file__), "FingerImages")
myList = os.listdir(folderPath)
print(myList)
overLayList = []
for imgPath in myList:
    image = cv2.imread(f'{folderPath}/{imgPath}')
    overLayList.append(image)
# print(len(overLayList)) ==> 6

while True:
    success, img = cap.read()

    #Detector things
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if(len(lmList))!=0:
        fingers = []

        #For the Thumb
        if lmList[tipIDS[0]][1] > lmList[tipIDS[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        #For the Fingers
        for id in range(1, 5):
            if lmList[tipIDS[id]][2] < lmList[tipIDS[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
    
        #Overlaying image
        totalFingers = fingers.count(1)
        print(totalFingers)
        overlayImg = cv2.resize(overLayList[totalFingers-1], (200, 200))  # Ensure size matches target region
        img[0:200, 0:200] = overlayImg #In python -1 means last index
        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
            10, (255, 0, 0), 25)

    #For Showing FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)

    cv2.imshow("Output", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break