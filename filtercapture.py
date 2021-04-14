import cv2 
import numpy as np

#get image classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')

filter = cv2.imread('mask.png')

  # creating 3 variables for the 3 attributes to be assigned to
orig_filter_h,orig_filter_w,filter_channels = filter.shape
gray_filter = cv2.cvtColor(filter, cv2.COLOR_BGR2GRAY)


ret, original_mask = cv2.threshold(gray_filter, 10, 255, cv2.THRESH_BINARY_INV)
original_mask_inv = cv2.bitwise_not(original_mask)

#read videoq
cap = cv2.VideoCapture(0)
ret, img = cap.read()
img_h, img_w = img.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))  # 20 fps

while True:   #continue to run until user breaks loop
    
    #read each frame of video and convert to gray
    ret, img = cap.read()


    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #find faces in image using classifier
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # get shape of img
    img_h,img_w,img_channels = img.shape

    #for each face
    for (x,y,w,h) in faces:

#coordinates of face region
        face_w = w
        face_h = h
        face_x1 = x
        face_x2 = face_x1 + face_w
        face_y1 = y
        face_y2 = face_y1 + face_h

        #filter size in relation to face by scaling
        filter_width = int(1.5 * face_w)
        filter_height = int(filter_width * orig_filter_h / orig_filter_w)
        #Test on filter
        #filter_height = int(filter_width * .5)
        
        #setting location of coordinates of filter
        filter_x1 = face_x2 - int(face_w/2) - int(filter_width/2)
        filter_x2 = filter_x1 + filter_width
        filter_y1 = face_y1 - int(face_h*.33)
        filter_y2 = filter_y1 + filter_height 

        #check to see if out of frame
        if filter_x1 < 0:
            filter_x1 = 0
        if filter_y1 < 0:
            filter_y1 = 0
        if filter_x2 > img_w:
            filter_x2 = img_w
        if filter_y2 > img_h:
            filter_y2 = img_h

        #Account for any out of frame changes
        filter_width = filter_x2 - filter_x1
        filter_height = filter_y2 - filter_y1

        #resize filter to fit on face
        filter = cv2.resize(filter, (filter_width,filter_height), interpolation = cv2.INTER_AREA)
        mask = cv2.resize(original_mask, (filter_width,filter_height), interpolation = cv2.INTER_AREA)
        mask_inv = cv2.resize(original_mask_inv, (filter_width,filter_height), interpolation = cv2.INTER_AREA)

        #take ROI for filter from background that is equal to size of filter image
        roi = img[filter_y1:filter_y2, filter_x1:filter_x2]

        #original image in background (bg) where filter is not present
        roi_bg = cv2.bitwise_and(roi,roi,mask = mask)
        roi_fg = cv2.bitwise_and(filter,filter,mask=mask_inv)
        dst = cv2.add(roi_bg,roi_fg)

        #put back in original image
        img[filter_y1:filter_y2, filter_x1:filter_x2] = dst

        break
    
    winname ="Video Capture"
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
    cv2.imshow(winname,img) #display image
    out.write(img)
    #if user pressed 'q' break
    if cv2.waitKey(1) == ord('q'): # 
        break;

cap.release() #turn off camera 
cv2.destroyAllWindows() #close all windowsq



