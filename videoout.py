import cv2

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    with open('/dev/fb0', 'rb+') as buf:
       buf.write(frame)
cap.release()
