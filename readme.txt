Contains Information for Filter app & General Notes

About

The filter app will overlay a filer on the faces of the pictured individuals.

IMAGES
    png are the filters
    jpg are test images



Github

    - Staging changes allows you to selectively add certain files to a commit while passing over the changes made in other files.

    - To commit and push to github from VS Code
        1 Click Source Control Icon
        2 Click the check mark aka commit
        3 Click the 3 elipses(...) near the checkmark
        4 Select Push from menu


VS Code Shortcuts

    #To move code block left right -->  ctrl +[ or ]


CV2 Notes

    # halo.shape = the ".shape" function will get the h,w,channel attributes of the halo shape Ex.(1202, 789, 3)
    # Height represents the number of pixel rows in the image or the number of pixels in each column of the image array.
    # Width represents the number of pixel columns in the image or the number of pixels in each row of the image array.
    # Number of Channels represents the number of components used to represent each pixel. 3- RGB

        orig_halo_h,orig_halo_w,halo_channels = halo.shape

    #Note: Use THRESH_BINARY_INV when image isalready on transparent background
    cv2.THRESH_BINARY if image has a white background
        ret, original_mask = cv2.threshold(gray_filter_img, 10, 255, cv2.THRESH_BINARY_INV)



