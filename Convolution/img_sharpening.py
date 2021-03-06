# Sharpening an image with python #

# for run from terminal
# python img_conv.py --image pokemon.png

# import the necessary packages
from skimage.exposure import rescale_intensity
import numpy as np
import cv2
#import argparse

def convolve(image, kernel):
	# grab the spatial dimensions of the image, along with
	# the spatial dimensions of the kernel
	(iH, iW) = image.shape[:2]
	(kH, kW) = kernel.shape[:2]

	# allocate memory for the output image, taking care to "pad"
	# the borders of the input image so the spatial size is not reduced
	pad = (kW - 1) // 2
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
	output = np.zeros((iH, iW), dtype="float32")

	# loop over the input image, "sliding" the kernel across
	# each (x, y)-coordinate from left-to-right and top to bottom
	for y in np.arange(pad, iH + pad):
		for x in np.arange(pad, iW + pad):
			# extract the ROI of the image by extracting the
			# *center* region of the current (x, y)-coordinates dimensions
			roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

			# perform the actual convolution by taking the element-wise
			# multiplicate between the ROI and the kernel, then summing the matrix
			k = (roi * kernel).sum()

			# store the convolved value in the output (x,y)-
			# coordinate of the output image
			output[y - pad, x - pad] = k

	# rescale the output image to be in the range [0, 255]
	output = rescale_intensity(output, in_range=(0, 255))
	output = (output * 255).astype("uint8")

	# return the output image
	return output

# construct a sharpening filter
kernel = np.array((
	[0, -1, 0],
	[-1, 5, -1],
	[0, -1, 0]), dtype="int")


# load the input image and convert it to grayscale
image = cv2.imread("pokemon.png")
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# apply the kernel to the grayscale image using both
print("[INFO] Applying {} kernel".format("Sharpen"))
convoleOutput = convolve(gray_img, kernel)


# show the output images
cv2.imshow("original", gray_img)
cv2.imshow("{} - convole".format("Sharpen"), convoleOutput)

