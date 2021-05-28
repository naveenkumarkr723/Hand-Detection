import cv2
import mediapipe as mp
import time


class hand_detector():
    def __init__(self,mode = False, maxhands = 2 ,detectionCon = 0.5,trackingCon = 0.5):
        self.mode = mode
        self.maxhands = maxhands
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpHand = mp.solutions.hands
        self.hands = self.mpHand.Hands(self.mode,self.maxhands,self.detectionCon ,self.trackingCon)
        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self,img,draw = True):
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_bgr)
        #print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handlms,self.mpHand.HAND_CONNECTIONS)
        return img


    def findPosition(self,img,handno = 0 ,draw = True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handno]

            for id,lm in enumerate(myhand.landmark):
                #print(id,lm)
                h,w,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                lmlist.append([id,cx,cy])
                #print(id,cx,cy)
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 255, 0), -1)
        return lmlist

def main():
    past_time = 0
    ctime = 0
    cap = cv2.VideoCapture(0)
    detector = hand_detector()
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

if __name__ == "__main__":
    main()