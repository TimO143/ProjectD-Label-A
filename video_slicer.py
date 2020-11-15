# import the necessary packages
from imutils.video import FileVideoStream
import cv2
import imutils
import os
import time
import numpy as np
from copy import copy
from imutils import paths
import imageImprover as improver
import random

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join(["face_detection_model", "deploy.prototxt"])
modelPath = os.path.sep.join(["face_detection_model", "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# grab the image paths and randomly shuffle them
videoPaths = sorted(list(paths.list_files("dataset2")))
fouten = 0

for videoPath in videoPaths:
    # get file name and extension.
    fileName = videoPath.split("\\")[1]
    Name = fileName.split(".")
    extension = Name[1]
    Name = Name[0]

    # initialize the video stream, then allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture(videoPath)
    time.sleep(2.0)
    turner = 0

    ret, image = vs.read()

    done = False
    while(not done):
        cv2.imshow("Image", image)
        key = cv2.waitKey(1)
        message = input("(y) if it is straight: ")
        if message == "y":
            done = True
        else:
            turner = turner + 1

        image = imutils.rotate(image, 90)

        # Release everything.
        cv2.destroyAllWindows()

    count = 0
    while True:

        if count > 0:
            ret, image = vs.read()

        if ret:

            turningCheck = 0
            while(turner > turningCheck):
                image = imutils.rotate(image, 90)
                turningCheck = turningCheck + 1

            # construct a blob from the image
            imageBlob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False)

            image = imutils.resize(image, width=600)
            (h, w) = image.shape[:2]

            # apply OpenCV's deep learning-based face detector to localize
            # faces in the input image
            detector.setInput(imageBlob)
            detections = detector.forward()

            image2 = copy(image)

            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections
                if confidence > 0.8:
                    count = count + 1
                    print("Face detected")
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    face = image[startY:endY, startX:endX]
                    try:
                        dirCheck = False
                        for path in os.listdir('dataset3'):
                            if path == Name:
                                dirCheck = True

                        if not dirCheck:
                            os.mkdir('dataset3' + "/" + Name)

                        cv2.imwrite("dataset3/%s/%s.%s" % (Name, count, "jpg"), cv2.resize(face, (224, 224)))
                        improver.Darker_image_preloaded('dataset3', Name, count, cv2.resize(face, (224, 244)))
                        improver.Brighter_image_preloaded('dataset3', Name, count, cv2.resize(face, (224, 244)))
                        print("Image written")
                    except:
                        print("Error image not saved")

                    # connect face and text
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(image2, (startX, startY), (endX, endY), (0, 0, 255), 2)

            cv2.imshow("Image", image2)
            key = cv2.waitKey(1) & 0xFF
        else:
            break

    vs.release()
    cv2.destroyAllWindows()