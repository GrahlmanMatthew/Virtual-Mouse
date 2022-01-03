import cv2
import time
#import autopy
import pyautogui
import numpy as np
import mediapipe as mp
from utils.handDetector import HandDetector

window_name = "Virtual Mouse - OpenCV & MediaPipe"
monitorWidth, monitorHeight = pyautogui.size()

videoCapture = cv2.VideoCapture(0)
videoWidth, videoHeight = int(videoCapture.get(3)), int(videoCapture.get(4))
handDetector = HandDetector(maxHands=1)

smoothBuffer = 5
virtualMonitorSize = 100
prevX, prevY = monitorWidth / 2, monitorHeight / 2 
currX, currY = prevX, prevY

while True:
    res, frame = videoCapture.read()
    frame = cv2.flip(frame, 1)  # flip image along y axis
    
    frame = handDetector.findHands(frame)
    lmList = handDetector.findPosition(frame)
    if not len(lmList) == 0:
        fingers = handDetector.fingersUp()
        indexFingerX, indexFingerY = lmList[8][-2:]
        middleFingerX, middleFingerY = lmList[12][-2:]
        
        # virtual scroll bar
        scrollbarStartX, scrollbarStartY = videoWidth - virtualMonitorSize, virtualMonitorSize
        scrollbarEndX, scrollbarEndY = videoWidth - virtualMonitorSize + 50, videoHeight - virtualMonitorSize
        cv2.rectangle(frame, (scrollbarStartX, scrollbarStartY), (scrollbarEndX, scrollbarEndY), (255, 255, 0), 2)
        midpointY = int(((scrollbarEndY - scrollbarStartY) / 2) + scrollbarStartY)
        cv2.line(frame, (scrollbarStartX, midpointY), (scrollbarEndX, midpointY), (255, 255, 0), 2)
        
        # virtual screen
        virtualStartX, virtualStartY = virtualMonitorSize, virtualMonitorSize
        virtualEndX, virtualEndY = videoWidth - virtualMonitorSize, videoHeight - virtualMonitorSize
        cv2.rectangle(frame, (virtualStartX, virtualStartY), (virtualEndX, virtualEndY), (255, 0, 255), 2)
                
        # Move mode -> index finger up, middle finger down
        if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0:
            cv2.circle(frame, (indexFingerX, indexFingerY), 15, (255, 0, 0), cv2.FILLED)
            
            if indexFingerX >= virtualEndX and indexFingerX <= scrollbarEndX:
                if indexFingerY >= scrollbarStartY and indexFingerY <= scrollbarEndY:
                    if indexFingerY <= midpointY:   pyautogui.scroll(pow(smoothBuffer, 2))
                    elif indexFingerY > midpointY:  pyautogui.scroll(-pow(smoothBuffer, 2))
                                    
            else:
                # Normalize coords & move mouse to new location
                newX = np.interp(indexFingerX, (virtualMonitorSize, videoWidth - virtualMonitorSize), (0, monitorWidth))
                newY = np.interp(indexFingerY, (virtualMonitorSize, videoHeight - virtualMonitorSize), (0, monitorHeight))
                currX, currY = prevX + (newX - prevX) / smoothBuffer, prevY + (newY - prevY) / smoothBuffer     # smoothen mouse transition
                pyautogui.moveTo(currX, currY)
                prevX, prevY = currX, currY
            
        # Left click mode -> index and middle finger are both up and are close to each other
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0:
            distance, mx, my = handDetector.findDistance(8, 12, frame)  # 8 (index tip), 12 (middle tip), mx, my -> midpoint values between index & middle finger
            if distance <= 40:
                cv2.circle(frame, (mx, my), 15, (0, 0, 255), cv2.FILLED)
                pyautogui.click(button='left')

        # Right click mode -> index and middle finger are both up and are close to each other
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
            distance, mx, my = handDetector.findDistance(8, 16, frame)  # 8 (index tip), 16 (ring tip)
            if distance <= 70:
                cv2.circle(frame, (mx, my), 15, (0, 255, 0), cv2.FILLED)
                pyautogui.click(button='right')
   
    cv2.imshow(window_name, frame)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    if cv2.waitKey(1) == ord('q'): break
    
videoCapture.release()
cv2.destroyAllWindows()