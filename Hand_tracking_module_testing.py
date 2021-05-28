import cv2
import HandTrackingModule as mp
import time


past_time = 0
ctime = 0
cap = cv2.VideoCapture(0)
detector = mp.hand_detector()
while cap.isOpened():
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findPosition(img)
    if len(lmlist)!=0:
        print(lmlist[4])

    ctime = time.time()
    fps = 1 / (ctime - past_time)
    past_time = ctime

    cv2.putText(img, f'FPS :{(int(fps))}', (400, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    cv2.imshow("hand tracking", img)
    cv2.waitKey(1)