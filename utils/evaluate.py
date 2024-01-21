from os.path import join, realpath, dirname
import sys
path = realpath(__file__)
sys.path.append(join(dirname(path), "../"))
import skimage.io as io
from pathlib import Path
import seaborn as sns
from scipy.spatial import distance
import numpy as np
from sklearn.metrics import recall_score
from utils import metrics
import pandas as pd
from matplotlib import pyplot as plt
from multiprocessing import Pool
def eval_front(gt_front_file, pred_front_file):
    gt_img = io.imread(gt_front_file)
    gt = (gt_img > 200).astype(int)
    gt_flat = gt.flatten()
    pred_img = io.imread(pred_front_file)
    pred = (pred_img > 200).astype(int)
    pred_flat = pred.flatten()
    scores = {}
    return scores

def eval_img(gt_file, pred_file, img_name, uncertainty_file=None):
    """
    Evaluate segmentaion Ground-Truth, Prediction pair
    using Dice coefficent, Euclidian distance, IOU, Sensitivity and Specificity
    :param gt_file: Ground truth segmentation
    :param pred_file: Predicted segmentation
    :param img_name: filename of image
    :param uncertainty_file: uncertainty file for uncertainty score (optional)
    :return: scores: dictionary with results ('image', 'dice', 'euclidian', 'IOU' 'specificity', 'sensitivity')
    """
    gt_img = io.imread(gt_file, as_gray=True)
    gt = (gt_img > 200).astype(int)
    pred_img = io.imread(pred_file, as_gray=True)
    pred = (pred_img > 200).astype(int)
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    scores = {}
    try:
        scores['image'] = img_name
        scores['dice']= metrics.dice_coefficient(gt_flat, pred_flat)
        scores['euclidian'] = distance.euclidean(gt_flat, pred_flat)
        scores['IOU'] = metrics.IOU(gt_flat, pred_flat)
        scores['specificity'] = metrics.specificity(gt_flat, pred_flat)
        scores['sensitivity'] = recall_score(gt_flat, pred_flat)
        if uncertainty_file:
            uncertainty_img =  io.imread(uncertainty_file, as_gray=True)
            uncertainty = uncertainty_img / 65535
            scores['uncertainty'] = uncertainty.mean()
    except Exception as e:
        raise type(e)(str(e) + " for img " + img_name)

    return scores

def eval_img_multiclass(gt_file, pred_file, img_name, uncertainty_file=None):
    """
    (DEFUNCT!)
    Evaluation of Multiclass Ground_truth, Prediction segmentation pair
    :param gt_file: Ground truth segmentation
    :param pred_file: Predicted segmentation
    :param img_name: filename of image
    :param uncertainty_file: uncertainty file for uncertainty score (optional)
    :return: scores: dictionary with results
    """
    gt_img = io.imread(gt_file, as_gray=True)
    gt_img[gt_img == 254] = 255
    pred_img = io.imread(pred_file, as_gray=True)

    gt_rock = (gt_img == 0).astype(int)
    gt_glacier = ((gt_img >  0) & (gt_img <= 150)).astype(int)
    gt_water = (gt_img > 150).astype(int)
    pred_rock = (pred_img == 0).astype(int)
    pred_glacier = ((pred_img > 0) & (pred_img <= 150)).astype(int)
    pred_water = (pred_img > 150).astype(int)
    scores = {}
    try:
        dice_water = metrics.dice_coefficient(gt_water, pred_water)
        dice_rock = metrics.dice_coefficient(gt_rock, pred_rock)
        dice_glacier = metrics.dice_coefficient(gt_glacier, pred_glacier)
        scores['image'] = img_name
        scores['dice-water']=  dice_water
        scores['dice-rock']=  dice_rock
        scores['dice-glacier']= dice_glacier
        scores['dice']= (dice_water + dice_rock + dice_glacier) / 3
        scores['euclidian'] = distance.euclidean(gt_img.flatten(), pred_img.flatten())
        scores['IOU'] = metrics.IOU(gt_img.flatten(), pred_img.flatten())
        #scores['specificity'] = metrics.specificity(gt_flat, pred_flat)
        #scores['sensitivity'] = recall_score(gt_flat, pred_flat)
        #if uncertainty_file:
        #    uncertainty_img =  io.imread(uncertainty_file, as_gray=True)
        #    uncertainty = uncertainty_img / 65535
        #    scores['uncertainty'] = uncertainty.mean()
    except Exception as e:
        raise type(e)(str(e) + " for img " + img_name)

    return scores


def evaluate(gt_path, pred_path, out_path=None, multi_class=False):
    """
    Evaluate the prediction masks using the Ground-truth
    :param gt_path:  path to the ground segmentation masks
    :param pred_path:  path to the predicted segmentation masks
    :param out_path: Where to write the Results to
    :param multi_class: If multiclass classification is used (not giving correct results!)
    :return: pandas Dataframe containing the results for each image
    """
    pred_path = Path(pred_path)
    gt_path = Path(gt_path)
    if not out_path:
        out_path = pred_path

    pred_files =[]
    gt_files = []
    img_names = []
    uncertainty_files = []
    for f in gt_path.glob("*.png"):
        gt_files.append(str(f))
        if "_zones.png" in f.name:
            basename = f.name[:f.name.rfind('_')]
        else:
            basename = f.stem
        img_names.append(basename + '.png')
        if Path(pred_path, basename + "_pred.png").exists():
            pred_files.append(str(Path(pred_path, basename + '_pred.png')))
        elif Path(pred_path, basename + ".png").exists():
            pred_files.append(str(Path(pred_path, basename + '.png')))
        else:
            raise FileNotFoundError(str(Path(pred_path, basename + '.png')) + " not found")
        if Path(pred_path, basename + "_uncertainty.png").exists():
            uncertainty_files.append(str(Path(pred_path, basename + '_uncertainty.png')))

    if len(pred_files) != len(gt_files) != len(img_names):
        raise AssertionError("Prediction and Ground truth set size does not match")

    if len(uncertainty_files) != 0 and len(uncertainty_files) != len(pred_files):
        raise AssertionError("Nr of Uncertainty images does not match Nr of Prediction images")

    # Evaluate each gt, prediction pair using multiprocessing
    p = Pool()
    if len(uncertainty_files) > 0:
        set = zip(gt_files, pred_files, img_names, uncertainty_files)
    else:
        set = zip(gt_files, pred_files, img_names)
    if multi_class:
        scores = p.starmap(eval_img_multiclass, set)
    else:
        scores = p.starmap(eval_img, set)

    scores = pd.DataFrame(scores)
    scores.to_pickle(Path(out_path, 'scores.pkl'))

    # Create summary report
    if multi_class:
        header ='Average Dice\tDice-Water\tDice-Rock\tDice-Glacier\tIOU\tEucl'
        report = (
                  str(np.mean(scores['dice'])) + '\t'
                + str(np.mean(scores['dice-water'])) + '\t'
                  + str(np.mean(scores['dice-rock'])) + '\t'
                  + str(np.mean(scores['dice-glacier'])) + '\t'
                  + str(np.mean(scores['IOU']))+ '\t'
                  + str(np.mean(scores['euclidian'])) )
    else:
        header ='Dice\tIOU\tEucl\tSensitivity\tSpecificitiy'
        report = (str(np.mean(scores['dice'])) + '\t'
                + str(np.mean(scores['IOU'])) + '\t'
                + str(np.mean(scores['euclidian'])) + '\t'
                + str(np.mean(scores['sensitivity'])) + '\t'
                + str(np.mean(scores['specificity'])))
    report += '\n'


    print(header)
    print(report)

    with open(str(Path(out_path, 'ReportOnModel.txt')), 'w') as f:
        f.write(header + '\n')
        f.write(report)

    return scores


def plot_history(history, out_file, xlim=None, ylim=None, title=None):
    """
    Creates plot of the training loss history
    :param history: keras model history
    :param out_file: output plot image file
    :param xlim: x-axis limits
    :param ylim: y-axis limits
    :param title: plot title
    """
    plt.figure()
    plt.rcParams.update({'font.size': 18})
    plt.plot( history['loss'], 'X-', label='training loss', linewidth=4.0)
    plt.plot(history['val_loss'], 'o-', label='val loss', linewidth=4.0)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    if title:
        plt.title(title)
    plt.legend(loc='upper right')
    plt.minorticks_on()
    plt.grid(which='minor', linestyle='--')
    plt.savefig(out_file, bbox_inches='tight', format='png', dpi=200)
    plt.show()

if __name__ == '__main__':
    path = Path('/home/andreas/glacier-front-detection/output/multiclass/front1_zangh_multiclass')

    test_path = Path('../datasets/front_detection_dataset/test/')
    evaluate(Path(test_path, 'masks'), path, multi_class=True)
    #test_path = Path('/home/andreas/glacier-front-detection/datasets/Jakobshavn_front_only/test')
    #history = pickle.load(open(next(path.glob('history*.pkl')), 'rb'))
    #history = pd.read_csv(Path(path,'masks_predicted_unet_Enze19_2_binary_crossentropy_monitor_val_loss/history.csv'))
    #plot_history(history, Path(path, 'loss_plot.png') , title='Set1')

    #evaluate(Path(test_path,'images'), Path(test_path,'masks'), path)

    #evaluate(Path(test_path, 'images'), Path(test_path, 'masks'), path)
    #test_path = Path('/home/andreas/glacier-front-detection/datasets/Jakobshavn_front_only/test')
    #evaluate(Path(test_path, 'images'), Path(test_path, 'masks'), '/home/andreas/glacier-front-detection/output_pix2pix_/output_Jakobshavn_pix2pix')
    #for d in Path('/home/andreas/glacier-front-detection/output_pix2pix_front_only').iterdir():
    #    if d.is_dir():
    #        test_path = Path('/home/andreas/glacier-front-detection/datasets/Jakobshavn_front_only/val/patches')
    #        evaluate(Path(test_path, 'masks'), d)
            #history = pickle.load(open(next(d.glob('history*.pkl')), 'rb'))
            #plot_history(history, Path(d, 'loss_plot.png')) # , xlim=(-10,130), ylim=(0,0.8))
