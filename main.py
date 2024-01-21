import argparse
import time
from pathlib import Path
import json

from tensorflow.keras.models import load_model
from layers.BayesDropout import  BayesDropout

from utils import helper_functions
from loss_functions import *
from tensorflow.keras.losses import *
from preprocessing.preprocessor import Preprocessor
from preprocessing import filter
from predict import predict
from utils import  evaluate
from train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Glacier Front Segmentation')

    parser.add_argument('--epochs', default=250, type=int, help='number of training epochs (integer value > 0)')
    parser.add_argument('--patience', default=30, type=int, help='how long to wait for improvements before Early_stopping')
    parser.add_argument('--batch_size', default=-1, type=int, help='batch size (integer value), if -1 set batch size according to available gpu memery')
    parser.add_argument('--patch_size', default=256, type=int, help='size of the image patches (patch_size x patch_size')

    parser.add_argument('--no_early_stopping', action='store_true',
                        help='Dont Use Early Stopping')
    parser.add_argument("--loss", help="loss function for the deep classifiers training ",
                        choices=["binary_crossentropy", "focal_loss", "combined_loss"], default="binary_crossentropy")
    parser.add_argument('--loss_parms', action=helper_functions.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...",
                        help='dictionary with parameters for loss function')
    parser.add_argument('--image_aug', action=helper_functions.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...",
                        help='dictionary with the augmentation for keras Image Processing',
                        default={'horizontal_flip': False, 'rotation_range': 0, 'fill_mode': 'nearest'})
    parser.add_argument("--denoise", help="Denoise filter",
                        choices=["none", "bilateral", "median", 'nlmeans', "enhanced_lee", "kuan"], default="None")
    parser.add_argument('--denoise_parms', action=helper_functions.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...",
                        help='dictionary with parameters for denoise filter')
    parser.add_argument('--patches_only', action='store_true', help='Training data is already split into image patches')

    parser.add_argument('--out_path', type=str, help='Output path for results')
    parser.add_argument('--data_path', type=str, help='Path containing training and val data')
    parser.add_argument('--model', default='two_stage', type=str, help='Training Model to use - can be pretrained model', choices=['unet', 'unet_bayes', 'two_stage'])
    parser.add_argument('--drop_rate', default=0.5, type=float, help='Dropout for Bayesian Unet')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--no_predict', action='store_true', help='Dont predict testset')
    parser.add_argument('--no_evaluate', action='store_true', help='Dont evaluate')
    parser.add_argument('--mc_iterations', type=int, default=20, help='Nr Monte Carlo Iterations for Bayes model')
    parser.add_argument('--second_stage', action='store_true', help='Second Stage training')
    parser.add_argument('--uncert_threshold', type=float, default=0.125, help='Threshold for uncertainty binarisation')
    parser.add_argument('--multi_class', action='store_true', help='Use MultiClass Segmentation')


    args = parser.parse_args()

    START = time.time()

    patch_size = args.patch_size

    batch_size = args.batch_size

    # check for training data path
    train_path = Path(args.data_path, 'train')
    if not Path(train_path, 'images').exists():
        if Path(train_path,'patches/images').exists() and args.model != 'two_stage':
            train_path = Path(train_path, 'patches')
        else:
            raise FileNotFoundError("training images Path not found")
    if len(list(Path(train_path, 'images').glob("*.png"))) == 0:
        raise FileNotFoundError("No training images were found")

    # Check for validation data path
    val_path = Path(args.data_path, 'val')
    if not Path(val_path, 'images').exists():
        if Path(val_path,'patches/images').exists():
            val_path = Path(val_path, 'patches')
        else:
            raise FileNotFoundError("Validation images Path not found")
    if len(list(Path(val_path, 'images').glob("*.png"))) == 0:
        raise FileNotFoundError("No validation images were found")

    # Check for test data path
    if not args.no_predict:
        test_path = Path(args.data_path, 'test')
        if not Path(test_path, 'images').exists():
            if Path(test_path,'patches/images').exists():
                test_path= Path(test_path, 'patches')
            else:
                raise FileNotFoundError("test images Path not found")
        if len(list(Path(test_path, 'images').glob("*.png"))) == 0:
            raise FileNotFoundError("No test images were found")

    # Set output path
    if args.out_path:
        out_path = Path(args.out_path)
    else:
        out_path = Path('data_' + str(patch_size) + '/test/masks_predicted_' + time.strftime("%y%m%d-%H%M%S"))

    if not out_path.exists():
        out_path.mkdir(parents=True)

    # log all arguments including default ones
    with open(Path(out_path, 'options.json'), 'w') as f:
        f.write(json.dumps(vars(args)))

    # Preprocessing
    preprocessor = Preprocessor()
    if args.denoise:
        preprocessor.add_filter(filter.get_denoise_filter(args.denoise, args.denoise_parms))

    # get loss function from function name
    loss_function = get_loss_function(args.loss, args.loss_parms)

    if 'bayes' in args.model or 'uncert' in args.model:
        mc_iterations = args.mc_iterations
    else: # set Nr iterations to 1 for regular u-net
        mc_iterations = 1

    # 2-Stage Optimization Process
    if args.model == 'two_stage':
        #1st Stage

        # if 1st stage already trained use trained model
        if Path(out_path, '1stStage', 'model_unet_bayes.h5').exists():
            model_1st = load_model(str(Path(out_path, '1stStage', 'model_unet_bayes.h5').absolute()), custom_objects={'loss': loss_function, 'BayesDropout':BayesDropout})

            if Path(out_path, '1stStage', 'options.json').exists():
                options_1st = json.load(open(Path(out_path, '1stStage', 'options.json'), 'r'))
            else:
                options_1st  = None

            if options_1st is not None and 'cutoff' in options_1st:
                cutoff_1st = options_1st['cutoff']
            elif args.multi_class:
                cutoff_1st = None
            else:
                cutoff_1st = 0.5
        else:
            print("1st stage training")
            model_1st, cutoff_1st = train('unet_bayes', train_path, val_path, Path(out_path, '1stStage'), args, loss_function=loss_function, preprocessor=preprocessor)
        print("1st stage prediction")
        predict(model_1st,
                Path(train_path, 'images'),
                Path(out_path, '1stStage', 'train'),
                batch_size=batch_size,
                patch_size=patch_size,
                preprocessor=preprocessor,
                cutoff=cutoff_1st,
                mc_iterations=args.mc_iterations)
        predict(model_1st,
                Path(val_path, 'images'),
                Path(out_path, '1stStage', 'val'),
                batch_size=batch_size,
                patch_size=patch_size,
                preprocessor=preprocessor,
                cutoff=cutoff_1st,
                mc_iterations=args.mc_iterations)

        ##2ndStage
        print("2nd stage training")
        model, cutoff = train('unet_bayes', train_path, val_path, out_path, args,
                                                   train_uncert_path=Path(out_path,'1stStage/train/uncertainty'),
                                                   val_uncert_path=Path(out_path, '1stStage/val/uncertainty'),
                                                   loss_function=loss_function, preprocessor=preprocessor, second_stage=True)

        # predict 1st Stage
        if not args.no_predict:
            print("2nd stage prediction")
            predict(model_1st,
                    Path(test_path, 'images'),
                    Path(out_path, '1stStage'),
                    batch_size=batch_size,
                    patch_size=patch_size,
                    preprocessor=preprocessor,
                    cutoff=cutoff_1st,
                    mc_iterations=args.mc_iterations)
        if not args.no_evaluate:
            evaluate.evaluate(Path(test_path, 'masks'), Path(out_path, '1stStage'))

        # predict 2ndStage
            predict(model,
                    Path(test_path, 'images'),
                    out_path,
                    uncert_path=Path(out_path,'1stStage/uncertainty'),
                    batch_size=batch_size,
                    patch_size=patch_size,
                    preprocessor=preprocessor,
                    cutoff=cutoff,
                    mc_iterations=args.mc_iterations)
            if not args.no_evaluate:
                evaluate.evaluate(Path(test_path, 'masks'), out_path)

    # single stage mode
    else:
        model, cutoff = train(args.model, train_path, val_path, out_path, args, loss_function=loss_function, preprocessor=preprocessor)
        if not args.no_predict:
            if args.second_stage:
                uncert_test_path = Path(test_path, 'uncertainty')
            else:
                uncert_test_path = None
            predict(model,
                    Path(test_path, 'images'),
                    out_path,
                    uncert_path=uncert_test_path,
                    batch_size=batch_size,
                    patch_size=patch_size,
                    preprocessor=preprocessor,
                    cutoff=cutoff,
                    mc_iterations=args.mc_iterations)
            evaluate.evaluate(Path(test_path, 'masks'), out_path)
            if not args.no_evaluate:
                evaluate.evaluate(Path(test_path, 'masks'), out_path)




    END = time.time()
    print('Execution Time: ', END - START)
