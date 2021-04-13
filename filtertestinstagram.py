# import the necessary packages
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import cv2


#get image classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')


def select_mask(img):

            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #filter image 
            filter_img = cv2.imread('vendetta.png')
            gray_filter_img = cv2.cvtColor(filter_img, cv2.COLOR_BGR2GRAY)

            ret, original_mask = cv2.threshold(gray_filter_img, 10, 255, cv2.THRESH_BINARY_INV)
            original_mask_inv = cv2.bitwise_not(original_mask)

            #find faces in image using classifier
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)


            # creating 3 variables for the 3 attributes to be assigned to
            orig_filter_img_h,orig_filter_img_w,filter_img_channels = filter_img.shape

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

                #filter_img size in relation to face by scaling
                filter_img_width = int(1.5 * face_w)
                filter_img_height = int(filter_img_width * orig_filter_img_h / orig_filter_img_w)
                #Test on filter_img
                #filter_img_height = int(filter_img_width * .5)
                
                #setting location of coordinates of filter_img
                filter_img_x1 = face_x2 - int(face_w/2) - int(filter_img_width/2)
                filter_img_x2 = filter_img_x1 + filter_img_width
                filter_img_y1 = face_y1 - int(face_h*.33)
                filter_img_y2 = filter_img_y1 + filter_img_height 

                #check to see if out of frame
                if filter_img_x1 < 0:
                    filter_img_x1 = 0
                if filter_img_y1 < 0:
                    filter_img_y1 = 0
                if filter_img_x2 > img_w:
                    filter_img_x2 = img_w
                if filter_img_y2 > img_h:
                    filter_img_y2 = img_h

                #Account for any out of frame changes
                filter_img_width = filter_img_x2 - filter_img_x1
                filter_img_height = filter_img_y2 - filter_img_y1

                #resize filter_img to fit on face
                filter_img = cv2.resize(filter_img, (filter_img_width,filter_img_height), interpolation = cv2.INTER_AREA)
                mask = cv2.resize(original_mask, (filter_img_width,filter_img_height), interpolation = cv2.INTER_AREA)
                mask_inv = cv2.resize(original_mask_inv, (filter_img_width,filter_img_height), interpolation = cv2.INTER_AREA)

                #take ROI for filter_img from background that is equal to size of filter_img image
                roi = img[filter_img_y1:filter_img_y2, filter_img_x1:filter_img_x2]

                #original image in background (bg) where filter_img is not present
                roi_bg = cv2.bitwise_and(roi,roi,mask = mask)
                roi_fg = cv2.bitwise_and(filter_img,filter_img,mask=mask_inv)
                dst = cv2.add(roi_bg,roi_fg)

                #put back in original image
                img[filter_img_y1:filter_img_y2, filter_img_x1:filter_img_x2] = dst



def select_image():
    # grab a reference to the image panels
    global panelA, panelB
    # open a file chooser dialog and allow the user to select an input
    # image
    path = filedialog.askopenfilename()

    if len(path) > 0:
            # load the image from disk, convert it to grayscale, and detect
            # edges in it
            img = cv2.imread(path)
            edged = cv2.imread(path)

            select_mask(img)
                        

            # OpenCV represents images in BGR order; however PIL represents
            # images in RGB order, so we need to swap the channels
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            edged = cv2.cvtColor(edged, cv2.COLOR_BGR2RGB)
            # convert the images to PIL format...
            img = Image.fromarray(img)



            #Replace with new image
            edged = Image.fromarray(edged)
            # ...and then to ImageTk format
            image = ImageTk.PhotoImage(img)
            edged = ImageTk.PhotoImage(edged)

                # if the panels are None, initialize them
    if panelA is None or panelB is None:
            # the first panel will store our original image
            panelA = Label(image=image)
            panelA.image = image
            panelA.pack(side="left", padx=10, pady=10)
            # while the second panel will store the edge map
            panelB = Label(image=edged)
            panelB.image = edged
            panelB.pack(side="right", padx=10, pady=10)
        # otherwise, update the image panels
    else:
            # update the pannels
            panelA.configure(image=image)
            panelB.configure(image=edged)
            panelA.image = image
            panelB.image = edged

            # initialize the window toolkit along with the two image panels
root = Tk()
panelA = None
panelB = None
# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
#photo = PhotoImage(file = r"icon.png")
btn = Button(root, text="Select an image", command=select_image)
#msk = Button(root, text="Select an image", image=photo, command=select_mask("icon.png"))
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
#msk.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")


# kick off the GUI
root.mainloop()