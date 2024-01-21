import json
from tensorflow.keras.models import load_model
import argparse
from loss_functions import *
from tensorflow.keras.losses import binary_crossentropy
from layers.BayesDropout import  BayesDropout
from preprocessing.preprocessor import Preprocessor
from preprocessing import filter
import numpy as np
import skimage.io as io
import cv2
from utils.metrics import dice_coefficient_cutoff
from utils.evaluate import evaluate#, evaluate_dice_only
from preprocessing.image_patches import extract_grayscale_patches, reconstruct_from_grayscale_patches
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
from pathlib import Path

def predict(model, img_path, out_path, uncert_path=None, uncert_threshold=None, batch_size=16, patch_size=256, cutoff=0.5, preprocessor=None, mc_iterations = 20):
    """

    :param model: trained keras model
    :param img_path: path of input images
    :param out_path: path for output predictions masks
    :param uncert_path: path of input uncertainty masks for 2nd Stage prediction
    :param uncert_threshold:  binarization threshold for uncertainty masks
    :param batch_size: batch size for model prediction
    :param patch_size: size the images are divided into before processing
    :param cutoff: binarization threshold for predictions
    :param preprocessor: preprocessing filter
    :param mc_iterations: Nr of Monte Carlo iterations for Bayesian U-Net
    """
    if not Path(out_path).exists():
        Path(out_path).mkdir(parents=True)

    output_channels = model.output_shape[-1]

    for filename in Path(img_path).rglob('*.png'):
        #print(filename)
        img = io.imread(filename, as_gray=True)
        img = img / 255
        if preprocessor is not None:
            img = preprocessor.process(img)
        img_pad = cv2.copyMakeBorder(img, 0, (patch_size - img.shape[0]) % patch_size, 0, (patch_size - img.shape[1]) % patch_size, cv2.BORDER_CONSTANT)
        p_img, i_img = extract_grayscale_patches(img_pad, (patch_size, patch_size), stride = (patch_size, patch_size))
        p_img = np.reshape(p_img,p_img.shape+(1,))

        # Read in uncertainty masks and divide them into patches
        if uncert_path is not None:
            if Path(uncert_path, filename.stem + '_uncertainty.png').exists():
                uncert = io.imread(Path(uncert_path, filename.stem + '_uncertainty.png'), as_gray=True)
            else:
                uncert = io.imread(Path(uncert_path, filename.stem + '.png'), as_gray=True)
            if preprocessor is not None:
                uncert = preprocessor.process(uncert)

            uncert = uncert / 65535
            if uncert_threshold is not None:
                uncert = (uncert >= uncert_threshold).astype(float)
            uncert_pad = cv2.copyMakeBorder(uncert, 0, (patch_size - uncert.shape[0]) % patch_size, 0, (patch_size - uncert.shape[1]) % patch_size, cv2.BORDER_CONSTANT)
            p_uncert, i_uncert = extract_grayscale_patches(uncert_pad, (patch_size, patch_size), stride = (patch_size, patch_size))
            p_uncert = np.reshape(p_uncert,p_uncert.shape+(1,))

            #combine image with uncertainty mask
            p_img = [np.concatenate((img, uncert), axis=2) for img, uncert in zip(p_img, p_uncert)]


        p_img_predicted  = []
        p_uncertainty = [] #output uncertainty patches
        for b_index in range((len(p_img) // batch_size) + 1):
            # Form image batches
            if ((b_index+1) * batch_size < len(p_img)):
                p_batch = np.array(p_img[b_index * batch_size:(b_index+1)*batch_size])
            elif b_index * batch_size < len(p_img):
                p_batch = np.array(p_img[b_index * batch_size:])
            else:
                break
            predictions = []

            # predictions as Monte Carlo iterations, for regular U-Net only one iteration is done
            for i in range(mc_iterations):
                prediction = model.predict(p_batch)
                predictions.append(prediction)
            p_img_predicted.append(np.mean(predictions, axis=0))
            p_uncertainty.append(np.var(predictions, axis=0))

        p_img_predicted = np.concatenate(p_img_predicted)
        p_uncertainty = np.concatenate(p_uncertainty)

        if output_channels > 1: # choose class with highest likelyhood
            p_img_predicted = np.argmax(p_img_predicted, axis=-1)
        else:
            p_img_predicted = p_img_predicted[..., 0]

        # restore full segmentation image from patches
        mask_predicted = reconstruct_from_grayscale_patches(p_img_predicted,i_img)[0]
        mask_predicted = mask_predicted[:img.shape[0], :img.shape[1]]

        # assign grayscale values to segmentation
        if output_channels > 1: # divide colorspace for multichannel output
            mask_predicted = (255 // output_channels) * mask_predicted
        elif cutoff is not None:
            # thresholding to make binary mask
            mask_predicted[mask_predicted < cutoff] = 0
            mask_predicted[mask_predicted >= cutoff] = 255
        else:
            mask_predicted = 255 * mask_predicted

        io.imsave(Path(out_path, filename.stem + '_pred.png'), mask_predicted.astype(np.uint8))


        if output_channels > 1:
            uncertainty = np.zeros(img.shape + (output_channels,))
            for ch in range(output_channels):
                mask_predicted = mask_predicted[:img.shape[0], :img.shape[1]]
                reconstructed = reconstruct_from_grayscale_patches(p_uncertainty[...,ch],i_img)[0]
                uncertainty[:,:, ch] = reconstructed[:img.shape[0], :img.shape[1]]
        else:
            uncertainty = reconstruct_from_grayscale_patches(p_uncertainty[..., 0],i_img)[0]
            uncertainty = uncertainty[:img.shape[0], :img.shape[1]]

        uncertainty_img = (65535 * uncertainty).astype(np.uint16)

        if not Path(out_path, 'uncertainty').exists():
            Path(out_path, 'uncertainty').mkdir()
        cv2.imwrite(str(Path(out_path, 'uncertainty', filename.stem + '_uncertainty.png')), uncertainty_img)


def get_cutoff_point(model, val_path, uncert_path=None, out_path=None, batch_size=16, patch_size=256, cutoff_pts=np.arange(0.0, 1.0, 0.05), preprocessor=None, mc_iterations=20, uncert_threshold=None):
    """

    :param model: pretrained keras model
    :param val_path: path to validation dataset
    :param out_path: optional output path for results and graphical plot
    :param batch_size: batch size for model prediction
    :param patch_size: size the images should be divided into
    :param cutoff_pts: range of threshold points to try
    :param preprocessor: preprocessing filters
    :param mc_iterations: Nr of Monter Carlo iterations for Bayesian U-Net
    :param uncert_threshold: For 2nd-Stage training: binarization threshold for uncertainty masks
    :return: cutoff_pt: best cutoff point found
             dice_all: the dice values for all tried cutoff points
    """
    if not 'bayes' in model.name:
        mc_iterations = 1
    if not Path(out_path).exists():
        Path(out_path).mkdir(parents=True)

    if not Path(val_path, 'images').exists() or len(list(Path(val_path, 'images').rglob('*.png'))) ==0:
        img_path = Path(val_path, 'patches/images')
        gt_path = Path(val_path, 'patches/masks')
        n_img = len(list(Path(img_path).rglob('*.png')))
        if n_img == 0:
            raise FileNotFoundError("No images found in " + val_path)
    else:
        img_path = Path(val_path, 'images')
        gt_path = Path(val_path, 'masks')
        n_img = len(list(Path(img_path).rglob('*.png')))

    dice_all = np.zeros(len(cutoff_pts))
    if uncert_path is None:
        uncert_path = Path(val_path, 'uncertainty')
    p = Pool()
    for filename in Path(img_path).rglob('*.png'):
        img = io.imread(filename, as_gray=True)
        img = img / 255
        if preprocessor is not None:
            img = preprocessor.process(img)
        img_pad = cv2.copyMakeBorder(img, 0, (patch_size - img.shape[0]) % patch_size, 0, (patch_size - img.shape[1]) % patch_size, cv2.BORDER_CONSTANT)
        p_img, i_img = extract_grayscale_patches(img_pad, (patch_size, patch_size), stride = (patch_size, patch_size))
        p_img = np.reshape(p_img,p_img.shape+(1,))

        if model.input_shape[3] == 2 and uncert_path is not None:
            if Path(uncert_path, filename.stem + '_uncertainty.png').exists():
                uncert = io.imread(Path(uncert_path, filename.stem + '_uncertainty.png'), as_gray=True)
            else:
                uncert = io.imread(Path(uncert_path, filename.stem + '.png'), as_gray=True)
            if preprocessor is not None:
                uncert = preprocessor.process(uncert)

            uncert = uncert / 65535
            if uncert_threshold is not None:
                uncert = (uncert >= uncert_threshold).astype(float)
            uncert_pad = cv2.copyMakeBorder(uncert, 0, (patch_size - uncert.shape[0]) % patch_size, 0, (patch_size - uncert.shape[1]) % patch_size, cv2.BORDER_CONSTANT)
            p_uncert, i_uncert = extract_grayscale_patches(uncert_pad, (patch_size, patch_size), stride = (patch_size, patch_size))
            p_uncert = np.reshape(p_uncert,p_uncert.shape+(1,))
            p_img = np.array([np.concatenate((img, uncert), axis=2) for img, uncert in zip(p_img, p_uncert)])


        predictions = []
        for i in range(mc_iterations):
            prediction = model.predict(p_img, batch_size=batch_size)
            predictions.append(prediction)
        # Only use mask channel
        predictions = np.array(predictions)[..., 0]

        p_img_predicted = predictions.mean(axis=0)
        mask_predicted = reconstruct_from_grayscale_patches(p_img_predicted,i_img)[0]
        mask_predicted = mask_predicted[:img.shape[0], :img.shape[1]]

        pred_flat = mask_predicted.flatten()

        if Path(gt_path, filename.stem + '_zones.png').exists():
            gt_img = io.imread(Path(gt_path, filename.stem + '_zones.png'), as_gray=True)
        else:
            gt_img = io.imread(Path(gt_path, filename.stem + '.png'), as_gray=True)
        gt = (gt_img > 200).astype(int)
        gt_flat = gt.flatten()
        dice_eval = partial(dice_coefficient_cutoff, gt_flat, pred_flat)
        dice_all += np.array(p.map(dice_eval, cutoff_pts))



    cutoff_pts_list = np.array(cutoff_pts)
    dice_all = np.array(dice_all) / n_img
    argmax = np.argmax(dice_all)
    cutoff_pt = cutoff_pts_list[argmax]
    max_dice = dice_all[argmax]

    if out_path is not None:
        #df = pd.DataFrame({'cutoff_pts':cutoff_pts_list, 'dice': dice_all})
        #df.to_pickle(Path(out_path, 'dice_cutoff.pkl')) # Save all values for later plot changes

        #plt.rcParams.update({'font.size': 18})
        #plt.figure()
        #plt.plot((cutoff_pt, cutoff_pt),(0, max_dice), linestyle=':', linewidth=2, color='grey')
        #plt.plot(cutoff_pts_list, dice_all)
        #plt.plot((cutoff_pt), (-0.02), ls="", marker="|", ms=10, color="k",
        #    clip_on=False, markeredgewidth=2)
        #plt.annotate(f'{max_dice:.2f}', (cutoff_pt, max_dice), fontsize='x-small')
        #plt.ylabel('Dice')
        #plt.xlabel('Cutoff Point')
        #plt.savefig(str(Path(out_path, 'cutoff.png')), bbox_inches='tight', format='png', dpi=200)

        plt.rcParams.update({'font.size': 18})
        fig, ax = plt.subplots()
        ax.set_ylim(0,1.00)
        ax.plot((cutoff_pt), (-0.02), ls="", marker="|", ms=10, color="k",
                clip_on=False, markeredgewidth=2)
        ax.plot((cutoff_pt, cutoff_pt),(0, max_dice), linestyle=':', linewidth=3, color='grey')
        ax.plot(cutoff_pts, dice_all, linewidth=5)
        ax.annotate(f'{max_dice:.2f}', (cutoff_pt, max_dice + 0.02))#, fontsize='x-small')
        ax.annotate(f'{cutoff_pt:.2f}', (cutoff_pt, 0.02), color='red')#, fontsize='x-small')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.ylabel('Dice')
        plt.xlabel('Decision Threshold')
        plt.savefig(str(Path(out_path, 'cutoff.png')), bbox_inches='tight', format='png', dpi=200)

    return cutoff_pt, dice_all

def predict_patches_only(model, img_path, out_path, uncert_path=None, uncert_threshold=None, batch_size=16, cutoff=0.5, preprocessor=None, mc_iterations = 20):
    """
    prediction optimized for small image patches
    :param model: trained keras model
    :param img_path: path of input images
    :param out_path: path of output predictions
    :param uncert_path: (optional) path of input uncertainty masks
    :param uncert_threshold: (optional) binarization threshold for uncertainty mask
    :param batch_size: batch size for model prediction
    :param cutoff: binarization threshold for segmentation predictions
    :param preprocessor: preprocessing filters
    :param mc_iterations: Nr of Monte Carlo iterations for Bayesian U-Net
    """
    if not Path(out_path).exists():
        Path(out_path).mkdir(parents=True)

    patches = []
    index = []
    for filename in Path(img_path).rglob('*.png'):
        #print(filename)
        img = io.imread(filename, as_gray=True)
        img = img / 255
        if uncert_path is not None:
            if Path(uncert_path, filename.stem + '_uncertainty.png').exists():
                uncert = io.imread(Path(uncert_path, filename.stem + '_uncertainty.png'), as_gray=True)
            else:
                uncert = io.imread(Path(uncert_path, filename.stem + '.png'), as_gray=True)

            uncert = uncert / 65535
            if uncert_threshold is not None:
                uncert = (uncert >= uncert_threshold).astype(float)
            img = np.stack((img, uncert), axis=-1)
        if preprocessor is not None:
            img = preprocessor.process(img)
        patches.append(img)
        index.append(filename.stem)
    for b_index in range((len(patches) // batch_size) +1):
        if (b_index+1 * batch_size < len(patches)):
            batch = np.array(patches[b_index * batch_size:(b_index+1)*batch_size])
        else:
            batch = np.array(patches[b_index * batch_size:])

        batch = np.reshape(batch,batch.shape+(1,))

        predictions = []
        for i in range(mc_iterations):
            prediction = model.predict(batch, batch_size=batch_size)
            predictions.append(prediction)
        predictions = np.array(predictions)
        patches_predicted = predictions.mean(axis=0)
        patches_uncertainty = predictions.var(axis=0)
        batch = np.reshape(batch, batch.shape[:-1])
        patches_predicted = patches_predicted.reshape(batch.shape)
        patches_uncertainty = patches_uncertainty.reshape(batch.shape)

        i = b_index * batch_size
        for patch, mask_predicted, uncertainty in zip(batch, patches_predicted, patches_uncertainty):
            patch = 255 * patch
            io.imsave(Path(out_path, index[i] + '.png'), patch.astype(np.uint8))


            if cutoff is not None:
                # thresholding to make binary mask
                mask_predicted[mask_predicted < cutoff] = 0
                mask_predicted[mask_predicted >= cutoff] = 255
            else:
                mask_predicted = 255 * mask_predicted

            io.imsave(Path(out_path, index[i] + '_pred.png'), mask_predicted.astype(np.uint8))

            uncertainty_img = (65535 * uncertainty).astype(np.uint16)
            io.imsave(Path(out_path, index[i] + '_uncertainty.png'), uncertainty_img, check_contrast=False )

            i += 1




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Glacier Front Segmentation Prediction')
    parser.add_argument('--model_path', type=str, help='Path containing trained model')
    parser.add_argument('--img_path', type=str, help='Path containing images to be segmented')
    parser.add_argument('--out_path', type=str, help='output path for predictions')
    parser.add_argument('--uncert_path', type=str, help='Path containing uncertainty images')
    parser.add_argument('--uncert_threshold', type=float, help='Threshold for uncertainty binarisation')
    parser.add_argument('--gt_path', type=str, help='Path containing the ground truth, necessary for evaluation_scripts')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size (integer value)')
    parser.add_argument('--cutoff', type=float, help='cutoff point of binarisation')
    parser.add_argument('--patches_only', action='store_true', help='optimized prediction algorithm for small image patches')
    args = parser.parse_args()

    if not Path(args.model_path).exists():
        print(args.model_path + " does not exist")
    if not Path(args.img_path).exists():
        print(args.img_path + " does not exist")
        exit(-1)
    if args.gt_path:
        if not Path(args.gt_path).exists():
            print(args.gt_path + " does not exist")
            exit(-1)
    if args.uncert_path:
        if not Path(args.uncert_path).exists():
            print(args.uncert_path + " does not exist")
            exit(-1)
        uncert_path = Path(args.uncert_path)
    else:
        uncert_path = None

    model_path = Path(args.model_path)
    options = json.load(open(Path(model_path, 'options.json'), 'r'))

    # Setup Preprocessing filters
    preprocessor = Preprocessor()
    if 'denoise' in options:
        if 'denoise_parms' in options:
            preprocessor.add_filter(filter.get_denoise_filter(options['denoise']))
        else:
            preprocessor.add_filter(filter.get_denoise_filter(options['denoise'], options['denoise_parms']))

    if 'contrast' in options and options['contrast']:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(25, 25))  # CLAHE adaptive contrast enhancement
        preprocessor.add_filter(clahe.apply)

    if 'loss_parms' in options:
        loss_function = get_loss_function(options['loss'], options['loss_parms'])
    else:
        loss_function = get_loss_function(options['loss'])

    model_file = next(model_path.glob('model_*.h5'))
    model_name = model_file.name[6:-3]
    model = load_model(str(model_file.absolute()), custom_objects={ 'loss': loss_function, 'BayesDropout':BayesDropout})
    multi_class = model.output_shape[-1] > 1
    print(model_name)


    if args.cutoff:
        cutoff = args.cutoff
    elif 'cutoff' in options:
        cutoff = options['cutoff']
    elif multi_class:
        cutoff = None
    else:
        cutoff = 0.5


    out_path = Path(args.out_path)

    if not out_path.exists():
        out_path.mkdir(parents=True)

    if args.patches_only:
        predict_patches_only(model,
                 args.img_path,
                 out_path, uncert_path,
                 batch_size=args.batch_size,
                 cutoff=cutoff,
                 preprocessor=preprocessor,
                 uncert_threshold=args.uncert_threshold)
    else:
        predict(model,
                args.img_path,
                out_path,uncert_path,
                batch_size=args.batch_size,
                patch_size=options['patch_size'],
                cutoff=cutoff,
                preprocessor=preprocessor)

    if args.gt_path:
        evaluate(args.gt_path, out_path, multi_class=multi_class)

