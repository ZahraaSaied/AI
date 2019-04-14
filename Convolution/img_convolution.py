# Implement image Convlution process with python 

# importing the liberaries
import numpy as np
import skimage
import matplotlib.pyplot as plt

img = skimage.io.imread("pokemon.png")
#plt.imshow(img)

img = skimage.color.rgb2gray(img)
#plt.imshow(img)

# Convlution filter
kernel = np.array([[0, -1, 0], 
                   [-1, 5, -1], 
                   [0, -1, 0]])

padded_img = np.zeros((img.shape[0]+2, img.shape[1]+2))
padded_img[1:padded_img.shape[0]-1, 1:padded_img.shape[1]-1] = img
img_map = np.zeros((padded_img.shape))

# Convlution process
for r in np.arange(1, padded_img.shape[0]-1):
    for c in np.arange(1, padded_img.shape[1]-1):
        img_region = padded_img[r-1:r+2, c-1:c+2]
        matching_result = img_region * kernel
        matching_value = np.sum(matching_result)
        img_map[r, c] = matching_value
        
output_img = img_map[1:img_map.shape[0]-1, 1:img_map.shape[1]-1]
plt.imshow(output_img).set_cmap("rgb")

