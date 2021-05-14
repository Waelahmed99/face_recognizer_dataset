import os
import cv2
import shutil
from os import environ
from colors import Colors  # Our color scheme


def main():
    """
    Capturing samples from device's camera to store it inside '/dataset/userName/'
    These samples are used as a dataset to recognize faces.
    Technique: Get user name, create a cascade classifier, capture rectangles and write it inside user's path.
    :return: void
    """
    suppress_qt_warnings()
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # video width
    cam.set(4, 480)  # video height
    face_detector = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

    # Get person id, or name.
    face_id = input(Colors.OK_BLUE + '\nenter user name/id:  ' + Colors.ENDC)
    print(Colors.HEADER + "Welcome, " + face_id)
    print("Initializing.. Please look at the camera.")

    # Capture up to [max_count] samples
    max_count = 30
    count = 0
    path = "dataset/" + face_id
    try:
        os.makedirs(path)
    except:
        pass

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            # Save the captured image into the datasets folder
            cv2.imwrite(path + "/User.".format(face_id) + str(face_id) + '.' +
                        str(count) + ".jpg", gray[y:y + h, x:x + w])
            cv2.imshow('image', img)
        key_press = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
        if key_press == 27:
            done = True
            try:
                shutil.rmtree(path)  # Remove directory if fails.
            except:
                pass
            print(Colors.FAIL + "Camera interrupt by user" + Colors.ENDC)
            break
        elif count >= max_count:  # Stop capturing after reaching [max_count]
            done = True
            print(Colors.OK_GREEN + "Done gathering {} samples".format(max_count))
            break

    if not done:
        print(Colors.FAIL + "an error occurred")
        try:
            shutil.rmtree(path)
        except:
            pass
    cam.release()
    cv2.destroyAllWindows()


def suppress_qt_warnings():
    """
    Ignores Deprecating warnings.
    :return: void.
    """
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"


if __name__ == "__main__":
    main()

