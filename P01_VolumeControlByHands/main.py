import cv2
import mediapipe as mp
import handTrackingModule as htm
import time
import math
import numpy as np
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# ======================== CONFIGURATION ========================
wCam, hCam = 648, 488  # Camera resolution
detector = htm.handDetector(detectionCon=0.75)  # Hand detector with confidence threshold
volBar = 400  # Initial volume bar height
pTime = 0  # For calculating FPS

# ======================== AUDIO SETUP ==========================
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
volRange = volume.GetVolumeRange()
minVol, maxVol = volRange[0], volRange[1]  # Usually -65.25 to 0.0

# ======================== CAMERA SETUP =========================
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# ======================== MAIN LOOP ============================
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if lmList:
        # Get coordinates of thumb tip (id 4) and index finger tip (id 8)
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw circles on fingertips and line between them
        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        # Calculate distance between fingers
        length = math.hypot(x2 - x1, y2 - y1)

        # Convert hand gesture distance to volume level
        vol = np.interp(length, [13, 190], [minVol, maxVol])
        volBar = np.interp(length, [13, 190], [400, 150])
        volume.SetMasterVolumeLevel(vol, None)

        # Visual feedback when fingers are too close or too far
        if length < 13 or length > 190:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    # Draw volume bar
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)

    # Display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 3)

    # Show result
    cv2.imshow("Hand Volume Control", img)
    cv2.waitKey(1)
