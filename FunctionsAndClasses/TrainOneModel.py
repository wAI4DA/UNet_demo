from FunctionsAndClasses.HEADER_torch import *
from FunctionsAndClasses.HEADER_utilities import *
from FunctionsAndClasses.HEADER_HRRR_URMA_Datasets_AllVars import *
from FunctionsAndClasses.HEADER_models import *
from FunctionsAndClasses.DefineModelAttributes import *
from FunctionsAndClasses.CONSTANTS import *

import torch.optim.lr_scheduler as lr_scheduler

C = CONSTANTS()

######################################################################################################################################################

def TrainOneModel(current_model_attrs,
                  resume_from_checkpoint=False,
                  checkpoint_model_attrs=None,
                  catch_loss_explosion=True,
                  INITIAL_LEARNING_RATE=1e-4,
                  NUM_GPUS_TO_USE=2, 
                  NUM_WORKERS=4,
                  TRAINING_LOG_FILEPATH = None,
                  TRAINED_MODEL_SAVEPATH = None
                 ):
    """
    Fully trains one model, whose attributes have already been defined before being fed to this function
    Defaults to using 2 GPUs (default per Ursa interactive node) but this is tunable with input params
    
    Inputs:
        - current_model_attrs = DefineModelAttributes object whose parameters have already been defined. 
            - MUST INVOKE THE FOLLOWING CLASS METHODS AHEAD OF TIME:
                - .create_dataset()
                - .set_model_architecture()
        - resume_from_checkpoint = bool to define if an existing model will be loaded from the input model's .savename and continue to be trained
            - Reads the # of epochs the model WAS trained for from current_model_attrs.NUM_EPOCHS, so make sure this was correctly set from set_model_attrs_from_savename() or manually set! This function will then continue to train for additional_epochs (e.g. if model was trained for 20 epochs, then called in this function with additional_epochs=80, the result will have been trained for, effectively, 100 epochs)
            - Sets current_model_attrs.NUM_EPOCHS to that+additional_epochs, so savename will be correct
            - TO DO (as of 2025-09-11): implement a better save than just the model weights; should include epoch #, optimizer weights as well
        - additional_epochs = int of the # of epochs to train for, if resume_from_checkpoint=True
        - catch_loss_explosion = bool to control if the model gets reverted if its loss explodes. Should generally be true, but manually set to False when training on experimental model architectures that .set_model_architecture can't handle
        - INITIAL_LEARNING_RATE = initial learning rate that the training will start with. Will be dynamically adjusted downwards by a factor of 0.1 if no improvement is seen for 10 epochs or ~10% of the requested # of epochs (whichever is lower)
        - NUM_GPUS_TO_USE = int for # GPUs to use with DataParallel and num_workers
        - NUM_WORKERS = int to set # workers per GPU. With patches dataset, should be set higher than 4 - be careful of exceeding the requested # of CPUs though!
        - TRAINING_LOG_FILEPATH = filepath to save training log to, including file name - should generally not be changed unless training multiple models simultaneously
        - TRAINED_MODEL_SAVEPATH = filepath to save trained models to - might need to differ if doing different losses, num epochs, etc
    """
    
    MULTIGPU_BATCH_SIZE = current_model_attrs.BATCH_SIZE*NUM_GPUS_TO_USE


    if resume_from_checkpoint:
        if checkpoint_model_attrs is not None:
            if current_model_attrs.NUM_EPOCHS > checkpoint_model_attrs.NUM_EPOCHS:
                additional_epochs = current_model_attrs.NUM_EPOCHS - checkpoint_model_attrs.NUM_EPOCHS
                upper_bound = additional_epochs+1
                print(f"Training from checkpoint for an additional {additional_epochs} epochs")
                print(f"Checkpoint model savename = {checkpoint_model_attrs.savename}")
                print(f"New model savename = {current_model_attrs.savename}")
                current_model_attrs.model.load_state_dict(torch.load(f"{TRAINED_MODEL_SAVEPATH}/{checkpoint_model_attrs.savename}.pt", weights_only=True))
                print(f"Model weights loaded from {checkpoint_model_attrs.savename}.pt")
            else:
                print(f"FATAL ERROR: new model needs to have a greater total number of epochs than the checkpoint model!")
        else:
            print(f"FATAL ERROR: checkpoint model's attributes were not initialized in the calling function!")
    else:
        upper_bound = current_model_attrs.NUM_EPOCHS+1
  
    device = torch.device("cuda")
    current_model_attrs.model = nn.DataParallel(current_model_attrs.model, device_ids=[i for i in range(NUM_GPUS_TO_USE)])
    current_model_attrs.model.to(device)
    
    if not os.path.exists(f"{TRAINED_MODEL_SAVEPATH}/{current_model_attrs.savename}.pt"):
        appendation = ""
    else: #model under that name already exists
        print(f"A model under {current_model_attrs.savename} already exists in the save directory! Change that model's name, or otherwise ensure non-overlapping names")
        appendation = "_nonoverlapping"
        print(f"Running script with model name {current_model_attrs.savename}{appendation}")
    
    with open(TRAINING_LOG_FILEPATH, "a") as file: 
        now = dt.datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"Starting {current_model_attrs.savename} | Current time = {current_time} \n")
    
    current_model_dataloader = DataLoader(current_model_attrs.dataset, 
                                          batch_size=MULTIGPU_BATCH_SIZE, 
                                          shuffle=True, 
                                          num_workers=NUM_WORKERS*NUM_GPUS_TO_USE, 
                                          pin_memory=True, 
                                          persistent_workers=True) #persistent_workers added 2025-08-25

    with open(TRAINING_LOG_FILEPATH, "a") as file:
        now = dt.datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"Data loaded | Current time = {current_time} \n")

    optimizer = torch.optim.AdamW(current_model_attrs.model.parameters(), lr=INITIAL_LEARNING_RATE, betas=[0.5,0.999]) 
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 
                                               factor=0.1, 
                                               patience=min(10, int(upper_bound/10)),
                                               threshold=1e-4)
                                               
    loss_function = torch.nn.L1Loss() 
    
    current_model_attrs.model.train()
    
    lowest_loss = 999
    previous_epoch_loss = 999
    consecutive_failure_counter = 0
    
    for epoch in range(1,upper_bound):

        epoch_loss = 0.0
        start = time.time()
        for i, (inputs,labels) in enumerate(current_model_dataloader):    
            start_batch = time.time()            
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            optimizer.zero_grad()
    
            outputs = current_model_attrs.model(inputs.float()) #weird datatype mismatching... for some reason it's seeing HRRR data as double
            loss = loss_function(outputs,labels)
            loss.backward()
            optimizer.step()
    
            epoch_loss += loss.item()

            divisor = (int(len(current_model_dataloader)/100))
            if divisor: #(i.e. divisor > 0, when len(dataloader) > 100)
                if (i+1)%divisor==0:
                    print(f"Done with batch {i+1}/{len(current_model_dataloader)}. Time = {time.time()-start_batch:.2f} sec. Running loss = {epoch_loss/(i+1):.5f}")
            else:
                if (i+1)%(int(len(current_model_dataloader)/10))==0:
                    print(f"Done with batch {i+1}/{len(current_model_dataloader)}. Time = {time.time()-start_batch:.2f} sec. Running loss = {epoch_loss/(i+1):.5f}")
            
        end = time.time() 
        print(f"End of epoch {epoch} | Average loss for epoch = {epoch_loss/len(current_model_dataloader):.5f} | Time for epoch = {end-start:.1f} sec | LR = {scheduler.get_last_lr()}")

        was_model_saved=False
        if epoch_loss <= lowest_loss: #only save models that have lower loss than previous best
            lowest_loss = epoch_loss
            torch.save(current_model_attrs.model.module.state_dict(), f"{TRAINED_MODEL_SAVEPATH}/{current_model_attrs.savename}{appendation}_TEMP.pt")
            print(f"New lowest loss - model saved @ epoch {epoch}")
            was_model_saved=True

        ################################################################################################################
        ### Measure to prevent explosions in loss from derailing training
        # If loss suddenly gets huge, stop, load the last saved model, revert epoch number by 1, go again
        if catch_loss_explosion and (epoch_loss > 100*previous_epoch_loss) and (epoch>1): #Works even in "False and False" case b/c Python only evaluates "True and True" as True here
            print(f"Epoch loss is >100x previous epoch's loss. Halting and loading last saved model")
            # Due to current_model_attrs.model now being in DataParallel, we can't load state dict directly
            # Horrible hack: initialize another identical DefineModelAttributes object, get that model loaded with the weights of the last checkpoint, put it in DataParallel, then hand that model to current_model_attrs.model
            tmp_model_attrs = DefineModelAttributes(current_model_attrs.is_train,
                                                    current_model_attrs.is_patches) #other attrs set from savename
            tmp_model_attrs.set_model_attrs_from_savename(f"{current_model_attrs.savename}{appendation}_TEMP")
            tmp_model_attrs.dataset = current_model_attrs.dataset
            tmp_model_attrs.num_channels_in = current_model_attrs.num_channels_in
            tmp_model_attrs.num_channels_out = current_model_attrs.num_channels_out
            tmp_model_attrs.set_model_architecture()
            tmp_model_attrs.set_model_weights(f"{current_model_attrs.savename}{appendation}_TEMP") #Note this will fail if the model explodes in the very first epoch - but I've never seen that occur, so it shouldn't be an issue
            tmp_model_attrs.model = nn.DataParallel(tmp_model_attrs.model, device_ids=[i for i in range(NUM_GPUS_TO_USE)])
            tmp_model_attrs.model.to(device)
            current_model_attrs.model = tmp_model_attrs.model
            
            consecutive_failure_counter+=1
            epoch = epoch-1
            del tmp_model_attrs 
        ################################################################################################################
        
        else:
            model_saved_str=""
            if was_model_saved:
                model_saved_str=f"| MODEL SAVED"
            
            with open(TRAINING_LOG_FILEPATH, "a") as file: 
                file.write(f"End of epoch {epoch} | Average loss for epoch = {epoch_loss/len(current_model_dataloader):.5f} | Time for epoch = {end-start:.1f} sec {model_saved_str} \n")
            
            consecutive_failure_counter = 0 #reset
            previous_epoch_loss = epoch_loss #only set this if the new epoch has not exploded; otherwise it'll log the exploded loss so the next epoch will always pass
            scheduler.step(epoch_loss)
            
        if consecutive_failure_counter > 10:
            print(f"Loss has exploded more than 10 consecutive times. Terminating training")
            sys.exit()

        
    with open(TRAINING_LOG_FILEPATH, "a") as file: 
        now = dt.datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"Training finished | Current time = {current_time} \n")
        
    os.rename(f"{TRAINED_MODEL_SAVEPATH}/{current_model_attrs.savename}{appendation}_TEMP.pt", 
              f"{TRAINED_MODEL_SAVEPATH}/{current_model_attrs.savename}{appendation}.pt") #so if training is interrupted, previously saved model under the same name isn't wiped out
    
    return
