import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime=0

#Media Pipe Classes
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection() # has a parameter of minConfidence

while True:
    sucess, img = cap.read()

    #convert to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    # print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection) #Prebuild funtion to do this
            # print(id, detection)
            # print(detection.score)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox  = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.width * ih), 
            cv2.rectangle(img, bbox, (255, 0, 255), 4)
            cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)


    #For showing FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)


    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Press 'q' to quit