import cv2

cap = cv2.VideoCapture("fd.mp4")

while True:
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    if not ret:
        break

    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)

    cv2.imshow("Motion Detection", dilated)

    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()
