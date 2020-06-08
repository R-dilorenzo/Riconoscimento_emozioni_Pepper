# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # Premendo il tasto ESC
        # Viene chiusa la finestra e terminata applicazione
        print "Premi ESC per chiudere la finestra" 
        break
    elif k%256 == 32:
        # Premendo il tasto SPACE
        # Viene salavato uno screenshot del frame di opencv nella directory corrente
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print "{} salvato!".format(img_name)
        img_counter += 1

cam.release()

cv2.destroyAllWindows()
