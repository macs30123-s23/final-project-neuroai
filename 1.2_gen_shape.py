"""
Generate Stimuli for Unit Probing

Grating Stimuli: Circle and triangle
Strategy:
    - Generate base image
    - Exapnd the base image to the background
    - move the slicing window across the image: window size = 224x224
    - moving step: 1
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

img_path = '/Users/anmin/Documents/Courses/2023Spring/Large_scale_computing\
/stimuli/shape'

def gen_circle():
    size = 112
    linewidth = 10
    circle_color = 'white'
    background_color = 'black'
    
    # Create a meshgrid of coordinates
    x = np.linspace(-size/2, size/2, size)
    y = np.linspace(-size/2, size/2, size)
    X, Y = np.meshgrid(x, y)

    # Calculate the distance from the center for each point
    radius = 40
    dist_from_center = np.sqrt(X**2 + Y**2)

    # Create the black background
    background = np.zeros((size, size))
    background_img = np.zeros((size, size))
    # Create the circle mask
    mask = np.abs(dist_from_center - radius) <= linewidth / 2
        
    # Set the circle color on the black background
    background[mask] = 1
    base_img = background.copy()

    # Expand the base image to the background
    row_1 = np.concatenate((base_img, background_img), axis=1)
    row_2 = np.concatenate((background_img, base_img), axis=1)
    base_1 = np.concatenate((row_1, row_2), axis=0)

    base_2 = np.concatenate((base_1, base_1), axis=0)
    base_3 = np.concatenate((base_2, base_2), axis=1)

    # move the slicing window across the image: window size = 224x224
    for i in range(0,224,10):
        for j in range(0,224,10):
            img_temp = base_3[i:i+224, j:j+224]
            plt.imshow(img_temp, cmap='gray')
            plt.axis('off')
            plt.savefig(os.path.join(img_path,
                                    'circle',
                                    f'{i}_{j}.png'), 
                        bbox_inches='tight', 
                        pad_inches=0
                        )
            plt.close()

def gen_triangle():
    # Set the size of the square and triangle properties
    size = 112
    linewidth = 10

    # Create a black background
    background = np.zeros((size, size, 3), dtype=np.uint8)

    # Define the vertices of the triangle
    side_length = 100
    height = int(np.sqrt(3) / 2 * side_length)  # Calculate the height of the equilateral triangle
    v1 = ((size - side_length) // 2, (size - height) // 2)
    v2 = ((size + side_length) // 2, (size - height) // 2)
    v3 = (size // 2, (size + height) // 2)

    # Draw the triangle on the black background
    cv2.line(background, v1, v2, (255, 255, 255), linewidth)
    cv2.line(background, v2, v3, (255, 255, 255), linewidth)
    cv2.line(background, v3, v1, (255, 255, 255), linewidth)

    #  convert image to numpy array
    base_img = np.array(background)

    # set stage 
    stage = np.zeros((size*4, size*4, 3), dtype=np.uint8)

    rotate_angle = [0, 90, 180, 270]
    for i in range(4):
        for j in range(4):
            # random pick rotate angle and rotate base image
            angle = rotate_angle[np.random.randint(0,4)]
            M = cv2.getRotationMatrix2D((size//2, size//2), angle, 1)
            img = cv2.warpAffine(base_img, M, (size, size))

            # Expand the base image to the background
            stage[size*i:size*(i+1), size*j:size*(j+1)] = img
    
    # move the slicing window across the image: window size = 224x224
    for i in range(0,224,10):
        for j in range(0,224,10):
            img_temp = stage[i:i+224, j:j+224]
            plt.imshow(img_temp, cmap='gray')
            plt.axis('off')
            plt.savefig(os.path.join(img_path,
                                    'triangle',
                                    f'{i}_{j}.png'), 
                        bbox_inches='tight', 
                        pad_inches=0
                        )
            plt.close()

def main():
    gen_circle()
    gen_triangle()

if __name__ == '__main__':
    main()

