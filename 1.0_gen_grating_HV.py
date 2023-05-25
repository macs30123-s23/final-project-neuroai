"""
Generate Stimuli for Unit Probing

Grating Stimuli: Horizontal and Vertical
Strategy:
    - Generate a large background image: 224x(224+1+40)
    - move the slicing window across the image: window size = 224x224
    - moving step: 1
"""
import numpy as np
import matplotlib.pyplot as plt
import os

img_path = '/Users/anmin/Documents/Courses/2023Spring/Large_scale_computing\
/stimuli/grating'

def gen_ver_hor_grating(is_vertical=True):
    # Parameters
    image_length, image_height = 224+1+40, 224

    # Generate a large black background image
    grating = np.zeros((image_length, image_height))
    # generate white gratings with length 5, step 32
    for i in range(0, image_length, 40):
        grating[i:i+5, :] = 1

    total_step = 39
    for i in range(total_step):
        grating_temp = grating[i:i+image_height, :]
        if is_vertical:
            grating_temp = grating_temp.T
            type_name = 'v'
        else:
            type_name = 'h'
        
        plt.imshow(grating_temp, cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(img_path,
                                 type_name,
                                 f'{i}.png'), 
                    bbox_inches='tight', 
                    pad_inches=0
                    )
        plt.close()

def main():
    gen_ver_hor_grating(is_vertical=True)
    gen_ver_hor_grating(is_vertical=False)

if __name__ == '__main__':
    main()

    