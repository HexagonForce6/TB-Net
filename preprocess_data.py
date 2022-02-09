
import numpy as np
import tensorflow as tf
import random
import cv2

RESIZE_SHAPE = (224, 224)

# Chance that a random augmentation will be performed on an image
AUGMENTATION_CHANCE = 1.0

# Create the average mask
avg_train_intensity = (0.531459229, 0.531459229, 0.531459229)

avg_mask = np.zeros((*RESIZE_SHAPE, 3))
black_mask = np.zeros((*RESIZE_SHAPE, 3))
for r in range(len(avg_mask)):
    for c in range(len(avg_mask[0])):
        # Hardcoded values for the corners.
        if r < 45 and (c < 45 or c > 178):
            avg_mask[r][c] = avg_train_intensity
            black_mask[r][c] = (0, 0, 0)
        else:
            black_mask[r][c] = (1, 1, 1)

'''
Detects if a row or column is padding.
If the row is comprised of the same value, from 0-5, or 250-255 then it is padding
'''
def is_padding(row):
    padding = False
    for i in range(6):
        if not np.count_nonzero(np.subtract(row,i)):
            padding = True
    for i in range(250, 256):
        if not np.count_nonzero(np.subtract(row,i)):
            padding = True
    return padding


'''
For the preprocessing of validation and testing data
'''
def preprocess_image_inference(image_path, image_size=RESIZE_SHAPE):
    '''
    Processes the image according to how the data was cleaned originally,
    and also how it is preprocessed in the data interface.
    '''

    # Read in the image as color, then split and use only the B channel
    image_color = cv2.imread(image_path, 1)
    b, g, r = cv2.split(image_color)
    height, width = b.shape
    image = b.copy()
    
    # Starting from the top, look at each row and mark those that are entirely black.
    # Stop once we reach a row with a non-black pixel. Repeat starting from the bottom.
    min_y = 0
    for row in image:
        if is_padding(row):
            min_y += 1
        else:
            break
    max_y = height - 1
    for row_index in range(height - 1, 0, -1):
        if is_padding(image[row_index]):
            max_y -= 1
        else:
            break

    # Do the same for columns, left to right and right to left
    min_x = 0
    # To iterate over columns, just iterate over the transpose of the image
    image_transposed = image.T
    for col in image_transposed:
        if is_padding(col):
            min_x += 1
        else:
            break
    max_x = width - 1
    for col_index in range(width - 1, 0, -1):
        if is_padding(image_transposed[col_index]):
            max_x -= 1
        else:
            break

    # Crop the image to these new boundaries
    image = image[min_y:max_y, min_x:max_x]
    # Create a 3 channel image by repeating the blue
    image = cv2.merge([image, image, image])
    # Resize the image
    image = cv2.resize(image, image_size, interpolation = cv2.INTER_LANCZOS4)

    # Normalize image, by rescaling the intensities to [0-1]
    min_value = np.min(b)
    max_value = np.max(b)
    image = np.divide(np.subtract(image, min_value), max_value)

    # Crop to smaller bounding box
    # We crop to the box (11, 11, 168, 202), then resize it back
    image = image[11:168, 11:202]
    image = cv2.resize(image, image_size, interpolation = cv2.INTER_LANCZOS4)

    # Apply the average mask in the upper corners.
    # This is done during training in order to hide possible metadata
    # that is included in some of the images. 
    image = cv2.bitwise_and(image, black_mask)

    # Renormalize the image 
    min_value = np.min(image)
    max_value = np.max(image)
    image = np.divide(np.subtract(image, min_value), max_value)

    # Finally, fill the corners with the average intensity of the training data
    image = cv2.addWeighted(image, 1, avg_mask, 1, 0)

    return image


'''
For the preprocessing and augmentation of training data
'''
def preprocess_image_train(image_path, image_size=RESIZE_SHAPE):

    # Read in the image as color, then split and use only the B channel
    image_color = cv2.imread(image_path, 1)
    b, g, r = cv2.split(image_color)
    height, width = b.shape
    image = b.copy()
    
    # Starting from the top, look at each row and mark those that are entirely black.
    # Stop once we reach a row with a non-black pixel. Repeat starting from the bottom.
    min_y = 0
    for row in image:
        if is_padding(row):
            min_y += 1
        else:
            break
    max_y = height - 1
    for row_index in range(height - 1, 0, -1):
        if is_padding(image[row_index]):
            max_y -= 1
        else:
            break

    # Do the same for columns, left to right and right to left
    min_x = 0
    # To iterate over columns, just iterate over the transpose of the image
    image_transposed = image.T
    for col in image_transposed:
        if is_padding(col):
            min_x += 1
        else:
            break
    max_x = width - 1
    for col_index in range(width - 1, 0, -1):
        if is_padding(image_transposed[col_index]):
            max_x -= 1
        else:
            break

    # Crop the image to these new boundaries
    image = image[min_y:max_y, min_x:max_x]
    # Create a 3 channel image by repeating the blue
    image = cv2.merge([image, image, image])
    # Resize the image
    image = cv2.resize(image, image_size, interpolation = cv2.INTER_LANCZOS4)

    # Normalize image, by rescaling the intensities to [0-1]
    min_value = np.min(b)
    max_value = np.max(b)
    image = np.divide(np.subtract(image, min_value), max_value)

    # Crop to smaller bounding box
    # We crop to the box (11, 11, 168, 202), then resize it back
    image = image[11:168, 11:202]
    image = cv2.resize(image, image_size, interpolation = cv2.INTER_LANCZOS4)

    # Chance for augmentation
    random.seed(2333333)
    if random.random() < AUGMENTATION_CHANCE:
        img = image.astype(np.float32)              # cvtColor does not support float64
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2 uses BGR, tf uses RGB
        img = tf.convert_to_tensor(img, dtype=tf.float32) 
        # Select an augmentation randomly to perform. randint is inclusive.
        which_aug = random.randint(0,3)
        if which_aug == 0:
            img = tf.image.random_crop(img, [202,202,3])
            img = tf.image.resize(img, [224,224])
        elif which_aug == 1:
            img = tf.image.random_flip_left_right(img)
        elif which_aug == 2:
            img = tf.image.random_brightness(img, 0.1)
        elif which_aug == 3:
            img = tf.image.random_contrast(img, 0, 0.2)

        image = tf.reverse(img, axis=[-1])      # convert from RGB back to BGR
        image = image.numpy().astype('float64') # convert back to float64

    # Apply the average mask in the upper corners.
    # This is done during training in order to hide possible metadata
    # that is included in some of the images. 
    image = cv2.bitwise_and(image, black_mask)

    # Renormalize the image 
    min_value = np.min(image)
    max_value = np.max(image)
    image = np.divide(np.subtract(image, min_value), max_value)

    # Finally, fill the corners with the average intensity of the training data
    image = cv2.addWeighted(image, 1, avg_mask, 1, 0)

    return image
