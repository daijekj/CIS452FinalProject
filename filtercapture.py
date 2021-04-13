import cv2 
import numpy as np

#get image classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')

halo = cv2.imread('vendetta.png')

  # creating 3 variables for the 3 attributes to be assigned to
orig_halo_h,orig_halo_w,halo_channels = halo.shape
# Haar Cascades and many facial recognition algorithms require images to be in grayscale
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_halo = cv2.cvtColor(halo, cv2.COLOR_BGR2GRAY)


ret, original_mask = cv2.threshold(gray_halo, 10, 255, cv2.THRESH_BINARY_INV)
original_mask_inv = cv2.bitwise_not(original_mask)

#read videoq
cap = cv2.VideoCapture(0)
ret, img = cap.read()
img_h, img_w = img.shape[:2]



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

        #halo size in relation to face by scaling
        halo_width = int(1.5 * face_w)
        halo_height = int(halo_width * orig_halo_h / orig_halo_w)
        #Test on halo
        #halo_height = int(halo_width * .5)
        
        #setting location of coordinates of halo
        halo_x1 = face_x2 - int(face_w/2) - int(halo_width/2)
        halo_x2 = halo_x1 + halo_width
        halo_y1 = face_y1 - int(face_h*.33)
        halo_y2 = halo_y1 + halo_height 

        #check to see if out of frame
        if halo_x1 < 0:
            halo_x1 = 0
        if halo_y1 < 0:
            halo_y1 = 0
        if halo_x2 > img_w:
            halo_x2 = img_w
        if halo_y2 > img_h:
            halo_y2 = img_h

        #Account for any out of frame changes
        halo_width = halo_x2 - halo_x1
        halo_height = halo_y2 - halo_y1

        #resize halo to fit on face
        halo = cv2.resize(halo, (halo_width,halo_height), interpolation = cv2.INTER_AREA)
        mask = cv2.resize(original_mask, (halo_width,halo_height), interpolation = cv2.INTER_AREA)
        mask_inv = cv2.resize(original_mask_inv, (halo_width,halo_height), interpolation = cv2.INTER_AREA)

        #take ROI for halo from background that is equal to size of halo image
        roi = img[halo_y1:halo_y2, halo_x1:halo_x2]

        #original image in background (bg) where halo is not present
        roi_bg = cv2.bitwise_and(roi,roi,mask = mask)
        roi_fg = cv2.bitwise_and(halo,halo,mask=mask_inv)
        dst = cv2.add(roi_bg,roi_fg)

        #put back in original image
        img[halo_y1:halo_y2, halo_x1:halo_x2] = dst

        break

    winname ="Video Capture"
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
    cv2.imshow(winname,img) #display image

    #if user pressed 'q' break
    if cv2.waitKey(1) == ord('q'): # 
        break;

cap.release() #turn off camera 
cv2.destroyAllWindows() #close all windowsq



