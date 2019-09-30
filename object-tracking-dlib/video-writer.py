# ------------------------
#   IMPORTS
# ------------------------
import numpy as np
import cv2

cap = cv2.VideoCapture("input/cameraimage_color.mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
isRecording = True

while (cap):
   # Capture frame-by-frame
   if(isRecording):
       ret, frame = cap.read()
   else:
       ret, frame = ret, frame

   # read the boolean to decide whether to write frame or not
   if(isRecording):
        out.write(frame)
   # Display the resulting frame
   cv2.imshow('video recording', frame)
   if cv2.waitKey(1) & 0xFF == ord('q'):
       break
   if cv2.waitKey(1) == 27:#Pause
       isRecording = False
   if cv2.waitKey(1) == 13:#Continue
       isRecording = True

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()