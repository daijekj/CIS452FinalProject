# Side note from article:
# Then, for each face, we have to place where the witch’s hat will go. Identify the coordinates for the boundaries of the face region using the height and width of the face. Then resize the witch image (or whatever filter you chose) to fit on the face region and select appropriate coordinates to place it on. You may have to fiddle around with these coordinates to get the image you chose to land it the right spot. In my witch example, I had to move up the filter/witch image, as seen below in witch_y1, because the brim of the hat should land on the person’s forehead. If I had not done that, the hat image would have been placed exactly where the face region was. Be sure to check to see if your filter image goes out of the frame of your main image. Lastly, use the masks to carve out the place to put your filter.


import cv2
import numpy as np

#path = '/Users/melis/anaconda3/Lib/site-packages/cv2/data'

#get image classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')
## Open CV
img = cv2.imread('me_small.jpg')
halo = cv2.imread('halo.png')

# Haar Cascades and many facial recognition algorithms require images to be in grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_halo = cv2.cvtColor(halo, cv2.COLOR_BGR2GRAY)

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
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    #select face as region of interest 
    roi_g = gray[y:y+h,x:x+h]
    roi_c = img[y:y+h,x:x+h]
    #within region of interest find eyes
    eyes = eye_cascade.detectMultiScale(roi_g)
    #for each eye
    for (ex,ey,ew,eh) in eyes:
        #draw retangle around eye
        cv2.rectangle(roi_c, (ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow('halo', halo)
cv2.imshow('img',img) #shows image
cv2.waitKey(0) #waits until a key is pressed to progress
cv2.destroyAllWindows() #closes windows
print(halo.shape)

#create mask and inverse mask of the halo
#Note: Use THRESH_BINARY_INV when image isalready on transparent background
#      cv2.THRESH_BINARY if image has a white background

ret, original_mask = cv2.threshold(gray_halo, 10, 255, cv2.THRESH_BINARY_INV)
original_mask_inv = cv2.bitwise_not(original_mask)