from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

class CustomEarlyStopping(EarlyStopping):
    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False,
                 prev_history=None):
        EarlyStopping.__init__(self,
                               monitor=monitor,
                               min_delta=min_delta,
                               patience=patience,
                               verbose=verbose,
                               mode=mode,
                               baseline=baseline,
                               restore_best_weights=restore_best_weights)

        self.prev_history = prev_history


    def on_train_begin(self, logs=None):
        if self.prev_history is not None:
            prev_history_monitor = self.prev_history[self.monitor]
            if self.monitor_op == np.less:
                best_arg = np.argmin(prev_history_monitor)
            else:
                best_arg = np.argmax(prev_history_monitor)

            self.wait = len(prev_history_monitor) - best_arg
            self.best = prev_history_monitor[best_arg]
            self.best_weights = self.model.get_weights()

            print(self.wait)

            if self.wait >= self.patience:
                self.model.stop_training = True
                print("Patience already reached")

        else:
            # Allow instances to be re-used
            self.wait = 0
            self.stopped_epoch = 0
            if self.baseline is not None:
              self.best = self.baseline
            else:
              self.best = np.Inf if self.monitor_op == np.less else -np.Inf
            self.best_weights = None

