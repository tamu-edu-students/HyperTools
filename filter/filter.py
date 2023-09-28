# import sys
# print(sys.path)
# import numpy as np
# from matplotlib import pyplot as plt
# import cv2

# # Load a grayscale image
# image = cv2.imread('cat.png', cv2.IMREAD_GRAYSCALE)

# # Check if the image was loaded successfully
# if image is None:
#     print("Image not found.")
# else:
#     # Apply filtering algorithm (e.g., Gaussian blur) to the image
#     kernel_size = (5, 5)  # Adjust kernel size as needed
#     sigma_x = 0           # Adjust sigma value as needed
#     filtered_image = cv2.GaussianBlur(image, kernel_size, sigma_x)

#     # Display the original and filtered images using Matplotlib
#     plt.subplot(121), plt.imshow(image, cmap='gray')
#     plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    
#     plt.subplot(122), plt.imshow(filtered_image, cmap='gray')
#     plt.title('Filtered Image'), plt.xticks([]), plt.yticks([])
    
#     plt.show()
