
from FunctionsAndClasses.HEADER_torch import *
from FunctionsAndClasses.HEADER_utilities import *
from FunctionsAndClasses.HEADER_FunctionsAndClasses import *
from FunctionsAndClasses.TrainOneModel_DDP import *

C = CONSTANTS()

###############

#Change these as needed - batch size is dependent on PATCH_SIZE in CONSTANTS
BATCH_SIZE = 360 #~300 for PATCH_SIZE=160, ~60 for PATCH_SIZE=400
NUM_EPOCHS = 25

TRAINING_LOG_FILEPATH = f"{C.DIR_UNET_MAIN}/training_log_mnode_mgpu_BS{BATCH_SIZE}_NE{NUM_EPOCHS}.txt"

###############

current_model_attrs = DefineModelAttributes(is_train=True,
                                            is_patches=True,
                                            is_western_domain=False,
                                            is_attention_model=True,
                                            is_residual_model=True,
                                            with_terrains=['diff'],
                                            predictor_vars=['u10m'],
                                            target_vars=['u10m'],
                                            BATCH_SIZE=BATCH_SIZE,
                                            NUM_EPOCHS=NUM_EPOCHS)

current_model_attrs.create_dataset()
current_model_attrs.set_model_architecture()

TrainOneModel_DDP(current_model_attrs,
                  INITIAL_LEARNING_RATE=2e-5,
                  NUM_WORKERS=10,
                  TRAINING_LOG_FILEPATH = TRAINING_LOG_FILEPATH,
                  TRAINED_MODEL_SAVEPATH = C.DIR_TRAINED_MODELS)
