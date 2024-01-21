import argparse
from os.path import join, realpath, dirname
import sys
path = realpath(__file__)
sys.path.append(join(dirname(path), "../"))
from pathlib import Path
import numpy as np
import cv2
from preprocessing import image_patches, preprocessor,augmentation
import json
import random
import pandas as pd
import string
from shutil import copy, rmtree

def process_imgs(in_dir, out_dir, patch_size=256, preprocessor = None, augment = None, border='zeros'):
    """
    Split images into patches
    :param in_dir: directory containing images
    :param out_dir: output path for image patches
    :param patch_size: size of the patches
    :param preprocessor: preprocessing filter
    :param augment: image augmentation function
    :param border: boundary rule: zeros or crop (default: zeros)
    :return: img_path_index: index information containing patchnumbers and original image name and size
    """

    if not Path(out_dir).exists():
        Path(out_dir).mkdir(parents=True)

    files_img = Path(in_dir).glob('*.png')

    patch_counter = 0
    img_patch_index = {}
    for f in files_img:
        basename = f.stem
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        shape = img.shape
        if preprocessor is not None:
            img = preprocessor.process(img)


        if border == 'zeros':
            img = cv2.copyMakeBorder(img, 0, (patch_size - img.shape[0]) % patch_size, 0, (patch_size - img.shape[1]) % patch_size, cv2.BORDER_CONSTANT)
        if border == 'crop':
            img = img[:img.shape[0] // patch_size, :img.shape[1] // patch_size]

        if augment is not None:
            imgs, augs = augment(img)
        else:
            imgs = [img]
            augs = ['']


        for img,augmentation  in zip(imgs, augs):

            p_img, i_img = image_patches.extract_grayscale_patches(img, (patch_size, patch_size), stride = (patch_size, patch_size))

            patch_indices = []
            for j in range(p_img.shape[0]):
                if np.count_nonzero(p_img[j])/(patch_size*patch_size) >= 0 and np.count_nonzero(p_img[j])/(patch_size*patch_size) <= 1:
                    cv2.imwrite(str(Path(out_dir, str(patch_counter)+'.png')), p_img[j])

                    patch_indices.append(patch_counter) # store patch nrs used for image
                    patch_counter += 1


            patch_meta_data = {}
            patch_meta_data['origin'] = [i_img[0].tolist(), i_img[1].tolist()]
            patch_meta_data['indices'] = patch_indices
            # Todo: Fix img shape for augmentations
            patch_meta_data['img_shape'] = list(shape)

            img_patch_index[basename+augmentation] = patch_meta_data


    with open(Path(out_dir, 'image_list.json'), 'w') as f:
        json.dump(img_patch_index, f)
    return img_patch_index


def process_data(in_dir, out_dir, patch_size=256, preprocessor = None, augment = None, front_zone_only=False, border='zeros', combine=False, uncert_minmax=False):
    """
    Split datafolder into patches
    :param in_dir: directory containing images, masks, uncertainty_masks folders
    :param out_dir: path to output directory
    :param patch_size: size of the patches
    :param preprocessor: preprocessing filter
    :param augment: image augmentation function
    :param front_zone_only: Only use patches containing both classes
    :param border: boundary rule: zeros or crop (default: zeros)
    :param combine: combine image and segmentation into single image
    :param uncert_minmax: Do minmax normalization on uncertainty masks
    :return: dictionary containing images and their patch numbers
    """

    if not Path(out_dir).exists():
        Path(out_dir).mkdir(parents=True)

    if not combine:
        if not Path(out_dir, 'images').exists():
            Path(out_dir, 'images').mkdir()
        if not Path(out_dir, 'masks').exists():
            Path(out_dir, 'masks').mkdir()

    if Path(in_dir, 'uncertainty').exists() and not Path(out_dir, 'uncertainty').exists():
        Path(out_dir, 'uncertainty').mkdir()

    if Path(in_dir, 'lines').exists() and not Path(out_dir, 'lines').exists():
        Path(out_dir, 'lines').mkdir()

    files_img = Path(in_dir, 'images').glob('*.png')

    patch_counter = 0
    img_patch_index = {}
    for f in files_img:
        basename = f.stem
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        shape = img.shape
        if preprocessor is not None:
            img = preprocessor.process(img)

        if Path(in_dir, 'masks', basename + '_zones.png').exists():
            mask_zones = cv2.imread(str(Path(in_dir, 'masks', basename + '_zones.png')), cv2.IMREAD_GRAYSCALE)
        else:
            mask_zones = cv2.imread(str(Path(in_dir, 'masks', basename + '.png')), cv2.IMREAD_GRAYSCALE)

        if Path(in_dir, 'lines').exists():
            if Path(in_dir, 'lines', basename + '_front.png').exists():
                front = cv2.imread(str(Path(in_dir, 'lines', basename + '_front.png')), cv2.IMREAD_GRAYSCALE)
            else:
                front = cv2.imread(str(Path(in_dir, 'lines', basename + '.png')), cv2.IMREAD_GRAYSCALE)

        if Path(in_dir, 'uncertainty').exists():
            if Path(in_dir, 'uncertainty', basename + '_uncertainty.png').exists():
                uncertainty = cv2.imread(str(Path(in_dir, 'uncertainty', basename + '_uncertainty.png')), cv2.IMREAD_GRAYSCALE)
            else:
                uncertainty = cv2.imread(str(Path(in_dir, 'uncertainty', basename + '.png')), cv2.IMREAD_GRAYSCALE)

            if uncert_minmax:
                uncertainty = 65535 * (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())
                uncertainty = uncertainty.astype(np.uint16)
        else:
            uncertainty = None

        if border == 'zeros':
            img = cv2.copyMakeBorder(img, 0, (patch_size - img.shape[0]) % patch_size, 0, (patch_size - img.shape[1]) % patch_size, cv2.BORDER_CONSTANT)
            mask_zones = cv2.copyMakeBorder(mask_zones, 0, (patch_size - mask_zones.shape[0]) % patch_size, 0, (patch_size - mask_zones.shape[1]) % patch_size, cv2.BORDER_CONSTANT)
            front = cv2.copyMakeBorder(front, 0, (patch_size - front.shape[0]) % patch_size, 0, (patch_size - front.shape[1]) % patch_size, cv2.BORDER_CONSTANT)
            if uncertainty is not None:
                uncertainty = cv2.copyMakeBorder(uncertainty, 0, (patch_size - uncertainty.shape[0]) % patch_size, 0, (patch_size - uncertainty.shape[1]) % patch_size, cv2.BORDER_CONSTANT)
        if border == 'crop':
            img = img[:img.shape[0] // patch_size, :img.shape[1] // patch_size]
            mask_zones = mask_zones[:img.shape[0] // patch_size, :img.shape[1] // patch_size]
            if uncertainty is not None:
                uncertainty= uncertainty[:img.shape[0] // patch_size, :img.shape[1] // patch_size]
            front = front[:img.shape[0] // patch_size, :img.shape[1] // patch_size]

        if augment is not None:
            imgs, augs = augment(img)
            masks_zones, _ = augment(mask_zones)
            uncertainties, _ = augment(uncertainty)
            fronts, _ = augment(front)
        else:
            imgs = [img]
            masks_zones = [mask_zones]
            uncertainties = [uncertainty]
            fronts = [front]
            augs = ['']


        for img, uncertainty, mask_zones, front,augmentation  in zip(imgs, uncertainties, masks_zones, fronts, augs):

            p_mask_zones, i_mask_zones = image_patches.extract_grayscale_patches(mask_zones, (patch_size, patch_size), stride = (patch_size, patch_size))
            p_img, i_img = image_patches.extract_grayscale_patches(img, (patch_size, patch_size), stride = (patch_size, patch_size))
            if uncertainty is not None:
                p_uncert, i_uncert = image_patches.extract_grayscale_patches(uncertainty, (patch_size, patch_size), stride = (patch_size, patch_size))
            p_front, i_front = image_patches.extract_grayscale_patches(front, (patch_size, patch_size), stride = (patch_size, patch_size))

            if front_zone_only:
                front_indices = []
                for i in range(p_mask_zones.shape[0]):
                    if 0 in p_mask_zones[i] and 254 in p_mask_zones[i] or 0 in p_mask_zones[i] and 255 in p_mask_zones[i]:
                        front_indices.append(i)
                front_indices = np.array(front_indices).astype(np.int)
                p_mask_zones = p_mask_zones[front_indices]
                i_mask_zones = (i_mask_zones[0][front_indices], i_mask_zones[1][front_indices])
                if uncertainty is not None:
                    p_uncert = p_uncert[front_indices]
                p_front = p_front[front_indices]
                p_img = p_img[front_indices]


            patch_indices = []
            for j in range(p_mask_zones.shape[0]):
                if np.count_nonzero(p_mask_zones[j])/(patch_size*patch_size) >= 0 and np.count_nonzero(p_mask_zones[j])/(patch_size*patch_size) <= 1:
                    if combine:
                        combined = np.concatenate((p_mask_zones[j], p_img[j]), axis=1)
                        cv2.imwrite(str(Path(out_dir, str(patch_counter)+'.png')), combined)
                    else:
                        cv2.imwrite(str(Path(out_dir, 'images/'+str(patch_counter)+'.png')), p_img[j])
                        cv2.imwrite(str(Path(out_dir, 'masks/'+str(patch_counter)+'.png')), p_mask_zones[j])
                    cv2.imwrite(str(Path(out_dir, 'lines/'+str(patch_counter)+'.png')), p_front[j])
                    if uncertainty is not None:
                        cv2.imwrite(str(Path(out_dir, 'uncertainty/'+str(patch_counter)+'.png')), p_uncert[j])

                    patch_indices.append(patch_counter) # store patch nrs used for image
                    patch_counter += 1


            patch_meta_data = {}
            patch_meta_data['origin'] = [i_mask_zones[0].tolist(), i_mask_zones[1].tolist()]
            patch_meta_data['indices'] = patch_indices
            # Todo: Fix img shape for augmentations
            patch_meta_data['img_shape'] = list(shape)

            img_patch_index[basename+augmentation] = patch_meta_data


    with open(Path(out_dir, 'image_list.json'), 'w') as f:
        json.dump(img_patch_index, f)

    return img_patch_index


def generate_subset(data_dir, out_dir, split=None, patch_size=256, preprocessor=None, augment=None, patches_only=False, border='zeros', front_zone_only=False, uncert_minmax=False):
    """
    Generate subset of dataset and generate image patches
    :param data_dir: directory containing images, masks, uncertainty_masks folders
    :param out_dir: path to output directory
    :param split: Ratio of data split
    :param patch_size: size of the patches
    :param preprocessor: preprocessing filter
    :param augment: image augmentation function
    :param patches_only: Only produce image patches
    :param border: boundary rule: zeros or crop (default: zeros)
    :param front_zone_only: Only use patches containing both classes
    :param uncert_minmax: Do minmax normalization on uncertainty masks
    """
    if not Path(data_dir).exists():
        print(str(data_dir) + " does not exist")


    files_img = list(Path(data_dir, 'images').glob('*.png'))

    if split is not None:
        if split < 1:
            img_subset = random.sample(files_img, int(split * len(files_img)))
        else:
            img_subset = random.sample(files_img, split)
    else:
        img_subset = files_img



    if not patches_only:
        if not Path(out_dir, 'images').exists():
            Path(out_dir, 'images').mkdir(parents=True)
        if not Path(out_dir, 'masks_zones').exists():
            Path(out_dir, 'masks_zones').mkdir()
        if Path(data_dir, 'uncertainty').exists() and not Path(out_dir, 'uncertainty').exists():
            Path(out_dir, 'uncertainty').mkdir()
        if Path(data_dir, 'masks_lines').exists() and not Path(out_dir, 'lines').exists():
            Path(out_dir, 'masks_lines').mkdir()

        for f in img_subset:
            print(f)
            basename = f.stem
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if preprocessor is not None:
                img = preprocessor.process(img)
            mask_zones = cv2.imread(str(Path(data_dir, 'masks_zones', basename + '_zones.png')), cv2.IMREAD_GRAYSCALE)
            if Path(data_dir, 'uncertainty').exists():
                uncertainty = cv2.imread(str(Path(data_dir, 'uncertainty', basename + '_uncertainty.png')), cv2.IMREAD_GRAYSCALE)
                if uncert_minmax:
                    uncertainty = 65535 * (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())
                    uncertainty = uncertainty.astype(np.uint16)
            else:
                uncertainty = None

            front = cv2.imread(str(Path(data_dir, 'lines', basename + '_front.png')), cv2.IMREAD_GRAYSCALE)


            if augment is not None:
                imgs, augs = augment(img)
                masks_zones, _ = augment(mask_zones)
                uncertainties, _ = augment(uncertainty)
                fronts, _ = augment(front)
            else:
                imgs = [img]
                masks_zones= [mask_zones]
                uncertainties = [uncertainty]
                fronts = [front]
                augs = ['']

            for img, uncertainty, mask_zones ,augmentation, front  in zip(imgs, uncertainties, masks_zones, augs, fronts):
                cv2.imwrite(str(Path(out_dir, 'images', basename + augmentation + '.png')), img)
                cv2.imwrite(str(Path(out_dir, 'masks_zones', basename + augmentation + '_zones.png')), mask_zones)
                cv2.imwrite(str(Path(out_dir, 'masks_lines', basename + augmentation + '_front.png')), front)
                if uncertainty is not None:
                    cv2.imwrite(str(Path(out_dir, 'uncertainty', basename + augmentation + '_uncertainty.png')), uncertainty)

    if patch_size is not None:
        process_data(data_dir,
                     Path(out_dir, 'patches'),
                     patch_size=patch_size,
                     preprocessor=preprocessor,
                     img_list=img_subset,
                     augment=augment,
                     front_zone_only=front_zone_only,
                     border=border,
                     uncert_minmax=uncert_minmax)



def split_set(data_dir, out_dir1, out_dir2, split):
    if not Path(out_dir1).exists():
        Path(out_dir1, 'images').mkdir(parents=True)
        Path(out_dir1, 'masks_zones').mkdir(parents=True)
        if Path(data_dir, 'uncertainty').exists():
            Path(out_dir1, 'uncertainty').mkdir(parents=True)
    if not Path(out_dir2).exists():
        Path(out_dir2, 'images').mkdir(parents=True)
        Path(out_dir2, 'masks_zones').mkdir(parents=True)
        if Path(data_dir, 'uncertainty').exists():
            Path(out_dir2, 'uncertainty').mkdir(parents=True)

    files_img = list(Path(data_dir, 'images').glob('*.png'))
    random.shuffle(files_img)
    if split < 1:
        split_point = int(split * len(files_img))
    else:
        split_point = split
    set1 = files_img[:split_point]
    set2 = files_img[split_point:]

    for f in set1:
        basename = f.stem
        copy(f, Path(out_dir1, 'images'))
        copy(Path(data_dir, 'masks_zones', basename + '_zones.png'), Path(out_dir1, 'masks_zones'))
        if Path(data_dir, 'uncertainty').exists():
            copy(Path(data_dir, 'uncertainty', basename + '_uncertainty.png'), Path(out_dir1, 'uncertainty'))

    for f in set2:
        basename = f.stem
        copy(f, Path(out_dir2, 'images'))
        copy(Path(data_dir, 'masks_zones', basename + '_zones.png'), Path(out_dir2, 'masks_zones'))
        if Path(data_dir, 'uncertainty').exists():
            copy(Path(data_dir, 'uncertainty', basename + '_uncertainty.png'), Path(out_dir2, 'uncertainty'))







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset Generator')
    parser.add_argument('--out_path', type=str, help='output path for dataset')
    parser.add_argument('--csv_file', type=str, help='Csv file containing img paths for dataset')
    parser.add_argument('--patch_size', type=int, default=256, help='Size of generated image patches')
    parser.add_argument("--seed", type=int, default=1, help="Seed for random number generator")

    args = parser.parse_args()
    random.seed(args.seed)
    out_path = Path(args.out_path)
    set = pd.read_csv(args.csv_file, usecols=['images', 'masks', 'lines'])
    if out_path.exists():
        rmtree(out_path)
    Path(out_path,'images').mkdir(parents=True)
    Path(out_path,'lines').mkdir(parents=True)
    Path(out_path,'masks').mkdir(parents=True)
    for index, row in set.iterrows():
        img = Path(row['images'])
        line = Path(row['lines'])
        mask = Path(row['masks'])

        img_out = Path(out_path, 'images', img.name)
        line_out = Path(out_path, 'lines', line.name)
        mask_out = Path(out_path, 'masks', mask.name)
        c = 0
        while (img_out.exists()):
            img_out = Path(out_path, 'images', img.stem + '_' + string.ascii_lowercase[c] + '.png')
            line_out = Path(out_path, 'lines', img.stem + '_' + string.ascii_lowercase[c] + '_front.png')
            mask_out = Path(out_path, 'masks', img.stem + '_' + string.ascii_lowercase[c] + '_zones.png')
            c += 1
        copy(img, img_out)
        copy(line, line_out)
        copy(mask, mask_out)

    process_data(out_path, Path(out_path, 'patches'), patch_size=args.patch_size)
