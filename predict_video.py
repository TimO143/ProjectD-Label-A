# import the necessary packages
from keras.models import load_model
from imutils.video import VideoStream
from imutils.video import FileVideoStream
import pickle
import cv2
import imutils
import os
import time
import numpy as np


# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join(["face_detection_model", "deploy.prototxt"])
modelPath = os.path.sep.join(["face_detection_model", "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load the model and label binarizer
print("[INFO] loading network and label binarizer...")
model = load_model("output/smallvggnet.model")
lb = pickle.loads(open("output/smallvggnet_lb.pickle", "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
#vs = FileVideoStream('test.mp4').start()
time.sleep(2.0)

# Timer to keep track of how long someone is recogniset
timer = time.time()

while True:
    image = vs.read()
    #image = imutils.rotate(image, 270)

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    image = imutils.resize(image, width=1400)

    #Checks if any good face is found in this iteration
    goodfaces = False

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # loop over the detections
    count2 = 0
    count3 = 0
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        count2 = count2 + 1

        # filter out weak detections
        if confidence > 0.8:
            # load the image, resize it to have a width of 600 pixels (while
            # maintaining the aspect ratio), and then grab the image dimensions
            (h, w) = image.shape[:2]

            # compute the (x, y)-coordinates of the bounding box for the
            # face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = image[startY:endY, startX:endX]
            height, width, channels = face.shape

            if(height > 64 and width > 64):
                face = cv2.resize(face, (64, 64))

                # scale the pixel values to [0, 1]
                face = face.astype("float") / 255.0

                face = face.reshape((1, face.shape[0], face.shape[1],
                                       face.shape[2]))

                # make a prediction on the image
                preds = model.predict(face)

                # find the class label index with the largest corresponding
                # probability
                i = preds.argmax(axis=1)[0]
                label = lb.classes_[i]

                precent = float(((preds[0][i]) * 100))

                # connect face and text
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)

                if(precent > 70 and "Person" not in label):
                    # draw the class label + probability on the output image
                    text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
                    cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    goodfaces = True

                    # Check if the recognition tackes longer then 5 seconds.
                    if(time.time() > timer + 5):
                        cv2.putText(image, "Open gate", (endX, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    cv2.putText(image, "Unknown", (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # If there is no recognition detected for this iteration or none detected people are recogniste then reset the timer.
    if(detections.shape[2] == 0 or not goodfaces):
        timer = time.time()

    # show the output image
    image = imutils.resize(image, width=600)
    cv2.imshow("Image", image)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
