import cv2
import time
import glob

# Set up the camera
video = cv2.VideoCapture(0)
# sleep to avoid the initial black colour in the matrix
time.sleep(1)

# We compare the first frame to the other frames to detect an object
first_frame = None
status_list = []
count = 1
while True:
    status = 0
    check, frame = video.read()
    # Converts to grayscale and then blurs the frames since we
    # dont need a precise picture. This will help reduce the data
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_gauss = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    if first_frame is None:
        first_frame = gray_gauss

    # The difference between the first frame and current frame
    delta_frame = cv2.absdiff(first_frame, gray_gauss)

    # We are going to take advantage of the matrix values and
    # Basically we are trying to convert the background as completely
    # black and the object which pops in as white.
    # THRESH_BINARY is because values can either be 0(Black) or 255(White)
    thresh_frame = cv2.threshold(delta_frame, 60, 255, cv2.THRESH_BINARY)[1]

    # reduces noise
    dil_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # Shows the frames to us
    cv2.imshow("video", dil_frame)

    # creates a contour array for objects
    contours, check = cv2.findContours(dil_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # iterate through each contour
    # if the contour is less than 12000 pixels, ignore it
    # else create a bounding rectangle for that object
    for contour in contours:
        if cv2.contourArea(contour) < 12000:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        rectangle = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        if rectangle.any():
            status = 1
            cv2.imwrite(f"images/{count}.png", frame)
            count += 1
            all_images = glob.glob("images/*.png")
            index = int(len(all_images)/2)
            img_object = all_images[index]
            


    status_list.append(status)
    status_list = status_list[-2:]

    if status_list[0] == 1 and status_list[1] == 0:
        print("email was sent")

    cv2.imshow("video", frame)

    # waits for the user to input a key
    key = cv2.waitKey(1)

    # if user types "q" then quits the video
    if key == ord("q"):
        break

video.release()



