import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

# Readin the color image: 'Emerald_Lakes_New_Zealand.jpg'
img = cv2.imread('Emerald_Lakes_New_Zealand_scr.jpg', cv2.IMREAD_COLOR)

# Convert to HSV (use: cvtColor())
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 

# Specify the lower and upper bound to segment the green lakes in the image.
g_lb = np.array([35, 50, 50], np.uint8)
g_ub = np.array([80, 255, 255], np.uint8)

# Create a green mask (use: inRange())
g_mask = cv2.inRange(img_hsv, g_lb, g_ub)
plt.subplot(132); plt.imshow(g_mask); plt.title('Green Mask')

# Segment the lakes (use bitwise_and())
g_seg = cv2.bitwise_and(img, img, mask = g_mask)
plt.subplot(132); plt.imshow(g_seg[:, :, ::-1]); plt.title('Green Color Segmented')
# Display the original image and the segmented lakes.
