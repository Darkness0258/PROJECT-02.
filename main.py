from PIL import Image, ImageEnhance
import cv2
import numpy as np


img = Image.open("FINAL PIC.jpg").convert("RGB")

img = img.resize((img.width*3, img.height*3), Image.BICUBIC)
cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
cv_img = cv2.fastNlMeansDenoisingColored(cv_img, None, 10, 10, 7, 21)
kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
cv_img = cv2.filter2D(cv_img, -1, kernel)
img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
img = ImageEnhance.Contrast(img).enhance(1.5)
img.save("enhanced_output.png")
