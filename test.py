import cv2

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
clf = cv2.CascadeClassifier(cascade_path)

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()

    if not ret:
        print("Failed to capture image from camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    n = 0

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        n = n + 1

    cv2.putText(frame, "Faces detected = " + str(n), (290, 64), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))

    cv2.imshow("Faces", frame)

    if cv2.waitKey(1) == ord("q"):
        break

print( "Faces detected = " + str(n))

camera.release()
cv2.destroyAllWindows()
