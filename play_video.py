import numpy as np
import cv2

while(True):
    # Capture frame-by-frame

    # Our operations on the frame come here


    # Display the resulting frame
    cv2.imshow('frame')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cv2.destroyAllWindows()