# Importing Libraries
import numpy as np
import cv2

# Global varriables used in draw_rectangle function
is_pressed = False
px = py = -1


def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img=img, center=(x, y), radius=30, color=(50, 50, 50), thickness=-1)


def draw_rectangle(event, x, y, flags, param):
    global img, is_pressed, px, py
    if event == cv2.EVENT_LBUTTONDOWN:
        is_pressed = True
        px, py = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if is_pressed:
            cv2.rectangle(img=img, pt1=(px, py), pt2=(x, y), color=(50, 50, 50), thickness=-1)

    elif event == cv2.EVENT_LBUTTONUP:
        cv2.rectangle(img=img, pt1=(px, py), pt2=(x, y), color=(50, 50, 50), thickness=-1)
        is_pressed = False


# cv2 window
cv2.namedWindow('canvas', cv2.WINDOW_AUTOSIZE)        # if WindowNormal is set, window can be resized
# Set Mouse Callback
# cv2.setMouseCallback('canvas', draw_circle)
cv2.setMouseCallback('canvas', draw_rectangle)

img = np.zeros((512, 512, 3), dtype=np.int8)

while True:
    '''
    k = cv2.waitKey(5) & 0xFF
    waitkey() -> It's argument is the time in milliseconds. 
                 he function waits for specified milliseconds for any keyboard event.
                 If you press any key in that time, the program continues.

    k -> ASCII value of the key pressed
    '''
    if cv2.waitKey(5) & 0xFF == 27:         # wait for ESC key to exit
        break
    cv2.imshow('canvas', img)

cv2.destroyAllWindows()
