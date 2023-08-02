import cv2
import face_recognition

# Load the image in BGR color space
imgElon = cv2.imread('download.jpg')

# Convert the image from BGRA to BGR if it has an alpha channel
if imgElon.shape[2] == 4:
    imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGRA2BGR)
# Perform face recognition or any other operations using the modified image


# Display the modified image
cv2.imshow('Elon Musk', imgElon)
cv2.waitKey(0)