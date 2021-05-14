from colors import Colors  # Our color scheme.
import face_recognition
from os import environ
import pickle
import cv2


def main():
    suppress_qt_warnings()
    face_cascade = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_alt2.xml")
    # load known faces from our face_enc binary file.
    try:
        data = pickle.loads(open('face_enc', "rb").read())
    except:
        print(Colors.HEADER + "Hi, it seems that this is your first run," + Colors.ENDC)
        print(Colors.OK_BLUE + "No data model is found inside project folder.")
        print("Please run analyze_face.py and analyze your face.")
        print("Then run save_faces.py to convert your images into binary file.")
        print(Colors.BOLD + Colors.OK_CYAN + "Thank you.")
        return

    print("Streaming started")
    video_capture = cv2.VideoCapture(0)

    while True:
        # get the frame from the threaded video stream
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60),
                                              flags=cv2.CASCADE_SCALE_IMAGE)

        # convert the input frame from BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb)
        names = []

        for encoding in encodings:
            # compare between faces in current stream, and saved faces from dataset.
            matches = face_recognition.compare_faces(data["encodings"],
                                                     encoding)
            name = "Unknown"
            # check to see if we have found a match
            if True in matches:
                # Find positions at which we get True and store them
                matched_ids = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matched_ids:
                    name = data["names"][i]
                    # increase count for the name we got
                    counts[name] = counts.get(name, 0) + 1
                # set name which has highest count
                name = max(counts, key=counts.get)

            # update the list of names
            names.append(name)
            # loop over the recognized faces
            for ((x, y, w, h), name) in zip(faces, names):
                # draw the predicted face name on the image
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
        key_press = cv2.waitKey(30) & 0xff
        if key_press == 27:  # press 'ESC' to quit
            print(Colors.FAIL + "Camera interrupt by user" + Colors.ENDC)
            break
    
    video_capture.release()
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


if __name__ == '__main__':
    main()
