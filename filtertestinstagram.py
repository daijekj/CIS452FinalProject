#To commit and push to github
#1 Click Source Control Icon
#2 Click the check mark aka commit
#3 Click the 3 elipses(...) near the checkmark
#4 Select Push from menu
import cv2
import numpy as np

#path = '/Users/melis/anaconda3/Lib/site-packages/cv2/data'

#get image classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')
## Open CV
img = cv2.imread('pic2.jpg')
halo = cv2.imread('witch.png')

# Haar Cascades and many facial recognition algorithms require images to be in grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_halo = cv2.cvtColor(halo, cv2.COLOR_BGR2GRAY)

#Note: Use THRESH_BINARY_INV when image isalready on transparent background
#      cv2.THRESH_BINARY if image has a white background
ret, original_mask = cv2.threshold(gray_halo, 10, 255, cv2.THRESH_BINARY_INV)
original_mask_inv = cv2.bitwise_not(original_mask)

#find faces in image using classifier
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# halo.shape = the ".shape" function will get the h,w,channel attributes of the halo shape
# creating 3 variables for the 3 attributes to be assigned to
orig_halo_h,orig_halo_w,halo_channels = halo.shape

# get shape of img
img_h,img_w,img_channels = img.shape

#for each face
for (x,y,w,h) in faces:
    #draw rectangle around face
   # img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)


        #coordinates of face region
    face_w = w
    face_h = h
    face_x1 = x
    face_x2 = face_x1 + face_w
    face_y1 = y
    face_y2 = face_y1 + face_h
    #within region of interest find eyes
    #eyes = eye_cascade.detectMultiScale(roi_g)
    #for each eye
     #halo size in relation to face by scaling
    halo_width = int(1.5 * face_w)
    halo_height = int(halo_width * orig_halo_h / orig_halo_w)
    
    #setting location of coordinates of halo
    halo_x1 = face_x2 - int(face_w/2) - int(halo_width/2)
    halo_x2 = halo_x1 + halo_width
    halo_y1 = face_y1 - int(face_h*1.25)
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


cv2.imshow('img',img) #display image
cv2.waitKey(0) #wait until key is pressed to proceed
cv2.destroyAllWindows() #close all windows
print(halo.shape)


# Last update 12:46am -- Error Message "inv_scale_x > 0 in function 'cv::resize'" - Possible data type issue
# Read Q&A below:
# https://stackoverflow.com/questions/55428929/error-while-resizing-image-error-215assertion-failed-func-0-in-functio

#Notes: 4/9/2021 **Add issue to documentation**
#The cv resize error message appears to do with the resizing of the halo object  === Line# 35 halo_width = int(.5 * face_w)
#The program runs with the witch hat width at halo_width = int(1.5 * face_w) but not int(.5 * face_w)
#Shape info of witch.png = (395, 360, 3) when running
#Need to do some math on the original shape size of the halo code function print(halo.shape)retunns the halo sizing
