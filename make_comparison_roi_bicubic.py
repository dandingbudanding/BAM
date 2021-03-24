import cv2

img=cv2.imread("./roi_bicubic/butterfly_.png")
h,w,_=img.shape
img=cv2.resize(img,(w//4,h//4),cv2.INTER_CUBIC)
img=cv2.resize(img,(w,h),cv2.INTER_CUBIC)

cv2.imwrite("./roi_bicubic/butterfly__bicubic.png",img)