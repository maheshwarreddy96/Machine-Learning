# Imports
import cv2
import glob
import numpy as np
import os
import pandas as pd
import shutil
import time

start = time.time()


def remove_image_from_list(_list, _image):
    """This function removes an image from a list. This is required because using the remove function for list performs
    element wise comparision of two arrays which is interpreted as being ambiguous. Thus we use Numpy's 'array_equal'
    function.'"""
    _index = 0
    while _index != len(_list) and not np.array_equal(_list[_index], _image):
        _index += 1
    if _index != len(_list):
        _list.pop(_index)
    else:
        raise ValueError('Image not found in list.')
    return _index


# Getting all the folder names in a list.
folder_name_list = []
for root, dirs, files in os.walk('./images'):
    for name in dirs:
        folder_name_list = dirs

print('List of folder names:', folder_name_list)

# Processing the images.

# Data-frame to store the image paths and their labels.
training_data_frame = pd.DataFrame(columns=['Path', 'Label'])
testing_data_frame = pd.DataFrame(columns=['Path', 'Label'])

# Creating a dictionary to store the images of different categories.
images = {}
for folder_name in folder_name_list:
    images[folder_name] = [cv2.imread(file) for file in glob.glob(f'./images/{folder_name}/*.jpg')][:500]
    i = 1

    # Rotating the images before resizing.
    for image in images[folder_name]:
        if image.shape[1] < image.shape[0]:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            index = remove_image_from_list(images[folder_name], image)
            images[folder_name].insert(index, rotated_image)

    # Resizing the images to the same resolution.
    for image in images[folder_name]:
        if image.shape[:2] != (224, 224):
            resized_image = cv2.resize(image, (224, 224))
            index = remove_image_from_list(images[folder_name], image)
            images[folder_name].insert(index, resized_image)

    # Deleting the original images.
    shutil.rmtree(f'./images/{folder_name}')

    # Replacing the original images with the resized images.
    os.makedirs(f'./images/{folder_name}')
    for image in images[folder_name]:
        cv2.imwrite(f'./images/{folder_name}/{i}.jpg', image)
        i += 1

    # Creating the labels for the images.
    for i in range(1, 401):
        training_data_frame = training_data_frame.append({'Path': f'./images/{folder_name}/{i}.jpg',
                                                          'Label': f'{folder_name}'}, ignore_index=True)
    for i in range(401, 501):
        testing_data_frame = testing_data_frame.append({'Path': f'./images/{folder_name}/{i}.jpg',
                                                        'Label': f'{folder_name}'}, ignore_index=True)
    images = {}
    print(f'Finished processing {folder_name}')

# Writing the csv file containing the paths and labels.
training_data_frame.to_csv('train_data_paths_and_labels.csv')
testing_data_frame.to_csv('test_data_paths_and_labels.csv')

end = time.time()
print(f'Runtime: {end - start}s')
