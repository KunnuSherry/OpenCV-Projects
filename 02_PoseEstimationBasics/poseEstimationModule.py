import cv2
import mediapipe as mp
import time

class PoseDetector():
    def __init__(self, mode=False, upperBody=False, smooth=True, detectCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upperBody = upperBody
        self.smooth = smooth
        self.detectCon = detectCon
        self.trackCon = trackCon
                
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
                    static_image_mode=self.mode,
                    model_complexity=1,
                    smooth_landmarks=self.smooth,
                    enable_segmentation=False,
                    min_detection_confidence=self.detectCon,
                    min_tracking_confidence=self.trackCon
                )        
        self.mpDraw = mp.solutions.drawing_utils
                
    def findPose(self, img, draw=True):
        img = cv2.resize(img, (640, 360))

        # Convert to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # results.pose_landmarks give cordinates
        self.results = self.pose.process(imgRGB)

        # Drawing the landmarks
        if draw:
            if self.results.pose_landmarks:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img


    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                print(id, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx,cy), 10, (255,0,0), cv2.FILLED)
        
        return lmList


    
def main():
    cap = cv2.VideoCapture(r'C:\Users\GIGABYTE\Desktop\OpenCVProjects\02_PoseEstimationBasics\TestingVideos\3.mp4')
    pTime=0
    detector = PoseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        print(lmList)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (78, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()