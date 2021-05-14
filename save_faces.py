from colors import Colors  # Our color scheme
from imutils import paths
import face_recognition
from os import environ
import pickle
import cv2
import os


def main():
    print(Colors.HEADER + "Dataset binary converter initializing..")
    image_paths = list(paths.list_images('dataset'))
    known_encodings = []
    known_names = []
    print("Done initializing, gathering dataset" + Colors.ENDC)
    print(Colors.UNDERLINE + "Note that folders inside dataset directory must contains person name." + Colors.ENDC)
    # loop over each path inside paths
    cnt = 0
    prev_name = ""
    for (i, imagePath) in enumerate(image_paths):
        name = imagePath.split(os.path.sep)[1]
        if name == prev_name:
            cnt += 1
        else:
            cnt = 1
            prev_name = name
            print('--------------')
        print(Colors.OK_BLUE + "Writing {}'s data [{}]".format(name, cnt))

        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Use Face_recognition to locate faces
        boxes = face_recognition.face_locations(rgb, model='hog')
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(name)

    # save encodings along with their names in dictionary data
    data = {"encodings": known_encodings, "names": known_names}
    # use pickle to save data into a file for later use (Binary file)
    f = open("face_enc", "wb")
    f.write(pickle.dumps(data))
    f.close()
    print(Colors.OK_GREEN + "Data converted and saved successfully.")


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

