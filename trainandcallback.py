# # IMPORTS
# FILE PATH MANAGEMENT IMPORT
import os
# BASE CALLBACK FOR SAVING MODELS IMPORT
from stable_baselines3.common.callbacks import BaseCallback
# CHECK ENVIRONMENT IMPORT
from stable_baselines3.common import env_checker


# # TRAIN AND LOGGING OF AI MODEL
# TRAINANDLOGGINGCALLBACK = SUBCLASS (CHILD)
# BASECALLBACK = SUPERCLASS (PARENT)
class TrainAndLoggingCallback(BaseCallback):
    # CHECK FREQ: HOW OFTEN WE WANT TO SAVE OUR MODEL
    # SAVE PATH: WHERE WE'D LIKE TO SAVE OUR MODEL INSTANCES
    # VERBOSITY LEVEL: 0 FOR NO OUTPUT, 1 FOR INFO MESSAGES, 2 FOR DEBUG MESSAGES
    def __init__(self, check_freq, save_path, verbose=1):
        # BELOW ANNOTATION USED IN PYTHON 2
        # SAME AS SUPER().__INIT__(VERBOSE)
        super(TrainAndLoggingCallback, self).__init__(verbose)
        # CHECK FREQ: HOW OFTEN WE WANT TO SAVE OUR MODEL
        self.check_freq = check_freq
        # SAVE PATH: WHERE WE'D LIKE TO SAVE OUR MODEL INSTANCES
        self.save_path = save_path

    # CREATE FOLDER IF NEEDED
    def _init_callback(self):
        # CHECK IF SAVE PATH HAS BEEN PROVIDED
        if self.save_path is not None:
            # MAKE DIRECTORY
            os.makedirs(self.save_path, exist_ok=True)

    # THIS METHOD WILL BE CALLED BY THE MODEL AFTER EACH TO CALL TO ENV.STEP()
    def _on_step(self):
        # IF THE NUMBER OF CALLS DIVIDED BY CHECK FREQUENCY IS EQUAL TO ZERO
        if self.n_calls % self.check_freq == 0:
            # SAVE MODEL
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True
