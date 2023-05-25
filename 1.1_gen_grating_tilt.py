"""
Generate Stimuli for Unit Probing

Grating Stimuli: tilted_45 and tilted_135
Strategy:
    - Generate a large background image 
    - rotate the background
    - move the slicing window
    - moving step: 1
"""
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.ndimage import rotate

img_path = '/Users/anmin/Documents/Courses/2023Spring/Large_scale_computing\
/stimuli/grating'

def gen_tilted_grating(angle=45):
    img_size = 224*2
    grating = np.zeros((img_size, img_size))
    for i in range(0, img_size, 40):
            grating[i:i+5, :] = 1
    
    if angle == 45:
        type_name = 'tilt_45'

        grating = rotate(grating, 45, reshape=False)
        # binary image
        grating[grating < 0.5] = 0
        grating[grating >= 0.5] = 1

        # find the upper left corner to start 
        for i in range(img_size):
            j = i 
            # find the first white pixel
            if grating[i, j] == 1:
                break
        
        # move the slicing window
        for k in range(int(40/(2**0.5))):
            grating_temp = grating[i+k:i+k+224, j+k:j+k+224]
            plt.imshow(grating_temp, cmap='gray')
            plt.axis('off')
            plt.savefig(os.path.join(img_path,
                                    type_name,
                                    f'{k}.png'), 
                        bbox_inches='tight', 
                        pad_inches=0
                        )
            plt.close()
    else:
        type_name = 'tilt_135'

        grating = rotate(grating, -45, reshape=False)
        # binary image
        grating[grating < 0.5] = 0
        grating[grating >= 0.5] = 1

        # find the upper left corner to start 
        for i in range(img_size):
            j = 447 - i
            # find the first white pixel
            if grating[i, j] == 1:
                break
        
        # move the slicing window
        for k in range(int(40/(2**0.5))):
            grating_temp = grating[i+k:i+k+224, j-k-224:j-k]
            plt.imshow(grating_temp, cmap='gray')
            plt.axis('off')
            plt.savefig(os.path.join(img_path,
                                    type_name,
                                    f'{k}.png'), 
                        bbox_inches='tight', 
                        pad_inches=0
                        )
            plt.close()

def main():
    gen_tilted_grating(angle=45)
    gen_tilted_grating(angle=135)

if __name__ == '__main__':
    main()
    