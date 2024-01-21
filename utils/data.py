from __future__ import print_function
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


def adjustData(img,mask,flag_multi_class):
    img = img / 255
    if(flag_multi_class):
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (3,))

        new_mask[mask == 0, 0] = 1
        new_mask[mask == 127, 1] = 1
        new_mask[mask == 254, 2] = 1
        new_mask[mask == 255, 2] = 1


        mask = new_mask
    else:
        mask = (mask > 127).astype(int)
    return (img,mask)

def trainGenerator(batch_size,train_path,image_folder, mask_folder, uncertainty_folder=None, front_folder=None,aug_dict=None,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,save_to_dir = None,target_size = (256,256),seed = 1, shuffle=True, uncert_threshold=None):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    if aug_dict == None:
        image_datagen = ImageDataGenerator()
        mask_datagen = ImageDataGenerator()
    else:
        image_datagen = ImageDataGenerator(**aug_dict)
        mask_datagen = ImageDataGenerator(**aug_dict)
        
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        shuffle=shuffle,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        shuffle=shuffle,
        seed = seed)
    if len(image_generator.filepaths) != len(mask_generator.filepaths):
        raise AssertionError("Different nr of input images and mask images")

    if uncertainty_folder is not None:
        if flag_multi_class:
            color_mode = 'rgb'
        else:
            color_mode = 'grayscale'
        uncertainty_generator= image_datagen.flow_from_directory(
            uncertainty_folder,
            classes = ['.'],
            class_mode = None,
            color_mode = color_mode,
            target_size = target_size,
            batch_size = batch_size,
            save_to_dir = save_to_dir,
            save_prefix  = image_save_prefix,
            shuffle=shuffle,
            seed = seed)
        if len(image_generator.filepaths) != len(uncertainty_generator.filepaths):
            raise AssertionError("Different nr of input images and uncertainty images")

    if uncertainty_folder is None:
        train_generator = zip(image_generator, mask_generator)
        for (img,mask) in train_generator:
            img,mask = adjustData(img,mask,flag_multi_class)
            yield (img,mask)
    else:
        train_generator = zip(image_generator, uncertainty_generator, mask_generator)
        for (img, uncertainty, mask) in train_generator:
            uncertainty = uncertainty / 65535
            if uncert_threshold is not None:
                uncertainty[uncertainty >= uncert_threshold] = 1
                uncertainty[uncertainty < uncert_threshold] = 0
            img, mask = adjustData(img,mask,flag_multi_class)
            combined = np.concatenate((img, uncertainty), axis=3)
            yield (combined,mask)
