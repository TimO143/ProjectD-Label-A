import cv2
import numpy as np
from imutils import paths
import os


def Brighter_image(main_write, write_location, Get_location):
    # Load imagePaths.
    imagePaths = sorted(list(paths.list_images(Get_location)))
    counter = 0

    # loop over the input images
    for imagePath in imagePaths:
        counter = counter + 1
        # load the image, resize it to 64x64 pixels (the required input
        # spatial dimensions of SmallVGGNet), and store the image in the
        # data list
        image = cv2.imread(imagePath)

        image = cv2.add(image, np.array([80.0]))
        dirCheck = False
        for path in os.listdir(main_write):
            if path == write_location:
                dirCheck = True

        if not dirCheck:
            os.mkdir(main_write + "/" + write_location)
        cv2.imwrite(main_write + "/" + write_location + "/" + str(counter) + "_light.jpg", image)

def Brighter_image_preloaded(main_write, write_location, name, image):

    image = cv2.add(image, np.array([80.0]))
    dirCheck = False
    for path in os.listdir(main_write):
        if path == write_location:
            dirCheck = True

    if not dirCheck:
        os.mkdir(main_write + "/" + write_location)
    cv2.imwrite(main_write + "/" + write_location + "/" + str(name) + "_light.jpg", image)


def Darker_image(main_write, write_location, Get_location):
    # Load imagePaths.
    imagePaths = sorted(list(paths.list_images(Get_location)))
    counter = 0

    # loop over the input images
    for imagePath in imagePaths:
        counter = counter + 1
        # load the image, resize it to 64x64 pixels (the required input
        # spatial dimensions of SmallVGGNet), and store the image in the
        # data list
        image = cv2.imread(imagePath)

        image = cv2.add(image, np.array([-80.0]))
        dirCheck = False
        for path in os.listdir(main_write):
            if path == write_location:
                dirCheck = True

        if not dirCheck:
            os.mkdir(main_write + "/" + write_location)
        cv2.imwrite(str(main_write) + "/" + str(write_location) + "/" + str(counter) + "_dark.jpg", image)

def Darker_image_preloaded(main_write, write_location, name, image):

    image = cv2.add(image, np.array([-80.0]))
    dirCheck = False
    for path in os.listdir(main_write):
        if path == write_location:
            dirCheck = True

    if not dirCheck:
        os.mkdir(main_write + "/" + write_location)
    cv2.imwrite(str(main_write) + "/" + str(write_location) + "/" + str(name) + "_dark.jpg", image)

if __name__ == '__main__':
    Paths = os.listdir("dataset")
    imagePaths = []

    for Path in Paths:
        if "Person" not in Path and "Reza" in Path:
            imagePaths.append(Path)

    images = []

    for imagePath in imagePaths:
        Darker_image("dataset2", imagePath, "dataset/" + imagePath)
        Brighter_image("dataset2", imagePath, "dataset/" + imagePath)
