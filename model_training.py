
from FunctionsAndClasses.HEADER_torch import *
from FunctionsAndClasses.HEADER_utilities import *
from FunctionsAndClasses.HEADER_FunctionsAndClasses import *

C = CONSTANTS()

##############################

#Change these as needed - batch size is dependent on PATCH_SIZE in CONSTANTS
BATCH_SIZE = 360 #~300 for PATCH_SIZE=160, ~60 for PATCH_SIZE=400
NUM_EPOCHS = 25

TRAINING_LOG_FILEPATH = f"{C.DIR_UNET_MAIN}/training_log_BS{BATCH_SIZE}_NE{NUM_EPOCHS}.txt"

##############################

current_model_attrs = DefineModelAttributes(is_train=True,
                                            is_patches=True,
                                            is_western_domain=False, #train over CONUS domain
                                            is_attention_model=True,
                                            is_residual_model=True,
                                            predictor_vars=['t2m'], #change this if you want to start with different variable(s)
                                            target_vars=['t2m'], #change this if you want to target different variable(s)
                                            with_terrains=['diff'], #generally, this should not be changed
                                            BATCH_SIZE=BATCH_SIZE,
                                            NUM_EPOCHS=NUM_EPOCHS)

current_model_attrs.create_dataset()
current_model_attrs.set_model_architecture()

TrainOneModel(current_model_attrs, 
              NUM_GPUS_TO_USE=2,
              NUM_WORKERS=10,
              INITIAL_LEARNING_RATE=2e-5, #Tweak this if the model isn't learning fast enough
              TRAINING_LOG_FILEPATH=TRAINING_LOG_FILEPATH, 
              TRAINED_MODEL_SAVEPATH=C.DIR_TRAINED_MODELS)
