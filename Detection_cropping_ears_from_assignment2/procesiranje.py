import glob
import cv2
import numpy as np
cv_img = []

# grem v vsako mapico proceis
# for img in glob.glob("C:/Users/sabin/Desktop/MAGISTERIJ/2_LETNIK/SB/IBB-Ear-Detection/oddaja/assignment3/testElon/*.png"):
# for img in glob.glob("C:/Users/sabin/Desktop/MAGISTERIJ/2_LETNIK/SB/IBB-Ear-Detection/oddaja/assignment3/testNatalie/*.png"):
for img in glob.glob("C:/Users/sabin/Desktop/MAGISTERIJ/2_LETNIK/SB/SB-assignmen3/data/trainElon/original/*.png"):
# for img in glob.glob("C:/Users/sabin/Desktop/MAGISTERIJ/2_LETNIK/SB/IBB-Ear-Detection/oddaja/assignment3/trainNatalie/*.png"):
    im = cv2.imread(img)

    #sharpening:
    sharpen = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
    im = cv2.filter2D(src=im, ddepth=-1, kernel=sharpen)

    #denoising
    im = cv2.fastNlMeansDenoisingColored(im, None, 10, 10, 7, 21)

    im = cv2.addWeighted(im, 0.8, im, 0, 1)

    cv_img.append(im)
    # cv2.imwrite('C:/Users/sabin/Desktop/MAGISTERIJ/2_LETNIK/SB/IBB-Ear-Detection/oddaja/assignment3/testElon/'+  img[-8:], im)
    # cv2.imwrite('C:/Users/sabin/Desktop/MAGISTERIJ/2_LETNIK/SB/IBB-Ear-Detection/oddaja/assignment3/testNatalie/'+  img[-8:], im)
    cv2.imwrite('C:/Users/sabin/Desktop/MAGISTERIJ/2_LETNIK/SB/SB-assignmen3/data/trainElon/processed/ELON_'+  img[-8:], im)
    # cv2.imwrite('C:/Users/sabin/Desktop/MAGISTERIJ/2_LETNIK/SB/IBB-Ear-Detection/oddaja/assignment3/trainNatalie/'+  img[-8:], im)




