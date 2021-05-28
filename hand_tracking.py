import cv2
import mediapipe as mp
import time

mpHand = mp.solutions.hands
hands = mpHand.Hands()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
past_time = 0
while cap.isOpened():
    success,img = cap.read()

    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_bgr)
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            for id,lm in enumerate(handlms.landmark):
                #print(id,lm)
                h,w,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                print(id,cx,cy)

                cv2.circle(img, (cx, cy), 10, (255, 255, 0), -1)
            mpDraw.draw_landmarks(img,handlms,mpHand.HAND_CONNECTIONS)

    ctime = time.time()
    fps = 1 / (ctime - past_time)
    past_time = ctime

    cv2.putText(img, f'FPS :{(int(fps))}', (400, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)

    cv2.imshow("hand tracking", img)
    cv2.waitKey(1)