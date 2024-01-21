from tensorflow.keras import backend as K
from tensorflow.keras.losses import *
import tensorflow as tf
from scipy.spatial.distance import dice

def focal_loss(alpha=1, gamma=1.5):
    def loss(y_true, y_pred):
        """
        Focal loss using the binary crossentropy implementation of keras as blueprint
        :param y_true:  True labels
        :param y_pred:  Predictions of the the same shape as y_true
        :return: loss value
        """
        epsilon = K.epsilon()
        y_pred_stable = K.clip(y_pred, epsilon, 1 - epsilon)
        weight_tp = (1 - y_pred_stable) ** gamma * y_true * K.log(y_pred_stable + epsilon)
        weight_tn = y_pred_stable ** gamma * (1 - y_true) * K.log(1 - y_pred_stable + epsilon)

        return - alpha * K.mean(weight_tp + weight_tn)

    return loss




def weighted_dice_loss(beta):
    def dice_loss(y_true, y_pred):
        numerator = 2 * K.sum(y_true * y_pred, axis=-1)
        denominator = K.sum(y_true + y_pred, axis=-1)

        return 1 - (numerator + 1) / (denominator + 1)


def combined_loss(loss_functions, split):

    def loss(y_true, y_pred):

        combined_result = 0
        for loss_function, weight in zip(loss_functions,split):
            combined_result += weight * loss_function(y_true, y_pred)

        return combined_result

    return loss


def get_loss_function(loss, loss_parameters=None):
    if loss == "combined_loss":
        if not loss_parameters:
            print("combined_loss needs loss functions as parameter")
        else:
            functions = []
            split = []
            for func_name, weight in loss_parameters.items():

                # for functions with additional parameters
                # generate loss function with default parameters
                # and standard y_true,y_pred signature
                if func_name == "focal_loss":
                    function = globals()[func_name]()
                else:
                    function = globals()[func_name]
                functions.append(function)
                split.append(float(weight))

            loss_function = combined_loss(functions, split)

    # for loss functions with additional parameters call to get function with y_true, y_pred arguments
    elif loss == 'focal_loss':
        if loss_parameters:
            loss_function = globals()[loss](**loss_parameters)
            #loss_function = locals()[loss_string](alpha=args.alpha, gamma=args.gamma)
        else:
            loss_function = globals()[loss]()
    elif loss == 'binary_crossentropy':
        loss_function = binary_crossentropy
    else:
        loss_function = globals()[loss]

    return loss_function
