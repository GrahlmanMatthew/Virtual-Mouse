import cv2
import math
import mediapipe as mp

class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands, min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        
        
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findDistance(self, id1, id2, img):
        x1, y1 = self.lmList[id1][1:]
        x2, y2 = self.lmList[id2][1:]
        dx, dy = (x1 + x2) // 2, (y1 + y2) // 2
        dist = math.hypot(x2 - x1, y2 - y1)
        return dist, dx, dy
    
    def fingersUp(self):
        fingerTipIds = [4, 8, 12, 16, 20]
        fingers = []
        for id in range(0, 5):
            if self.lmList[fingerTipIds[id]][2] < self.lmList[fingerTipIds[id]-1][2]: 
                fingers.append(1)
            else:   
                fingers.append(0)
        return fingers
        
    def findPosition(self, img, handNo = 0, draw = True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
        return self.lmList