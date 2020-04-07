import cv2
import argparse
import glob
import matplotlib.pyplot as plt


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str,default="input", help='input_image_folder')
    parser.add_argument('-o', type=str,default="output", help="second image to compare ")
    args = parser.parse_args()
    return args

def get_images(path):
    return glob.glob(f"{path}/*.png")

def get_faces(gray_image):
    faces = haar_classifier.detectMultiScale(gray_image, 1.3, 5)
    faces_detected = format(len(faces)) + " faces detected!"
    print(faces_detected)
    return faces

def draw_rectangles(faces,image):
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 5)
    return image

if __name__ == "__main__":
    args = parser()
    input_folder = args.i
    output_folder = args.o
    image_paths = get_images(input_folder)

    for i,path in enumerate(image_paths):
        print(path)
        haar_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        image = cv2.imread(path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = get_faces(gray_image)
        image = draw_rectangles(faces,image)
        image_rgb_faces = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imsave(f"{output_folder}/output{i}.png",image_rgb_faces)
