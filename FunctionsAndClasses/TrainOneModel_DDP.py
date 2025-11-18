from FunctionsAndClasses.HEADER_torch import *
from FunctionsAndClasses.HEADER_utilities import *
from FunctionsAndClasses.HEADER_HRRR_URMA_Datasets_AllVars import *
from FunctionsAndClasses.HEADER_models import *
from FunctionsAndClasses.DefineModelAttributes import *
from FunctionsAndClasses.CONSTANTS import *

import torch.optim.lr_scheduler as lr_scheduler

C = CONSTANTS()

######################################################################################################################################################

def TrainOneModel_DDP(current_model_attrs,
                      resume_from_checkpoint=False,
                      checkpoint_model_attrs=None,
                      catch_loss_explosion=False,
                      INITIAL_LEARNING_RATE=1e-4, 
                      NUM_WORKERS=4,
                      TRAINING_LOG_FILEPATH = None,
                      TRAINED_MODEL_SAVEPATH = None
                 ):
    """
    Fully trains (across multiple nodes) one model, whose attributes have already been defined before being fed to this function
    
    Inputs:
        - current_model_attrs = DefineModelAttributes object whose parameters have already been defined. 
            > MUST INVOKE THE FOLLOWING CLASS METHODS AHEAD OF TIME:
                - .create_dataset()
                - .set_model_architecture()
        - resume_from_checkpoint = bool to define if an existing model will be loaded from the input model's .savename and continue to be trained
            > Reads the # of epochs the model WAS trained for from current_model_attrs.NUM_EPOCHS, so make sure this was correctly set from set_model_attrs_from_savename() or manually set! This function will then continue to train for the difference between the checkpointed model's # of epochs (from its savename, so make sure that's accurate) and the new models # of epochs, which must be greater than the checkpointed model's. e.g. if checkpoint model was trained for 20 epochs and the new model has NUM_EPOCHS=100, then an additional 100-20 = 80 epochs will be done
            > TO DO (as of 2025-09-11): implement a better save than just the model weights; should include epoch #, optimizer weights as well
        - catch_loss_explosion = bool to control if the model gets reverted if its loss explodes. Should generally be True, but manually set to False when training on experimental model architectures that .set_model_architecture can't handle
            > (2025-10-24) NOT IMPLEMENTED FOR DDP - not sure how to deal with it in DDP, need to do some research/testing
        - INITIAL_LEARNING_RATE = initial learning rate that the training will start with. Should be reduced by LR scheduler after some # of plateaued epochs.
            > (2025-10-24) NOT IMPLEMENTED FOR DDP - not sure how to deal with it in DDP, need to do some research/testing
        - NUM_WORKERS = int to set # workers per GPU. With patches dataset, should be set higher than 4 - be careful of exceeding the requested # of CPUs though!
        - TRAINING_LOG_FILEPATH = filepath to save training log to, including file name - should generally not be changed unless training multiple models simultaneously
        - TRAINED_MODEL_SAVEPATH = filepath to save trained models to - might need to differ if doing different losses, num epochs, etc
    """
    
    if catch_loss_explosion:
        print(f"catch_loss_explosion not yet implemented for DDP. Setting to False")
        catch_loss_explosion = False


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

    #### Here begins DDP

    # Get SLURM info for DDP
    world_size = int(os.environ.get("WORLD_SIZE"))
    world_rank = int(os.environ.get("SLURM_PROCID", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    num_nodes = int(os.environ.get("SLURM_NNODES"))
    ntasks_per_node = int(os.environ.get("SLURM_NTASKS_PER_NODE")) #, 2))

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank() #does this conflict with, or is the same as, local_rank?
    torch.cuda.set_device(local_rank)
    
    print(f"local_rank = {local_rank}")
    if rank==0:
        print(f"[Rank {rank}] world_size = {world_size} | world_rank = {world_rank} | num_nodes = {num_nodes} | ntasks_per_node = {ntasks_per_node}")
    
    if rank==0:
        print(f"[Rank {rank}] Starting distributed training with {world_size} processes.") 
    
    # Model
    current_model_attrs.model.cuda()
    current_model_attrs.model = DDP(current_model_attrs.model, device_ids=[local_rank])

    if rank==0:
        print(f'[Rank {rank}] Model put onto DDP')
    
    
    # Dataset and DataLoader
    num_workers = min(NUM_WORKERS, os.cpu_count() // world_size) # dist.get_world_size()) #10
    print(f"[Rank {rank}] num_workers = {num_workers}")
    
    sampler = DistributedSampler(current_model_attrs.dataset,
                                 num_replicas=world_size, #from Raj's code
                                 #rank=world_rank, #from Raj's code; maybe should be local_rank or rank? #(2025-10-24) Commenting this out - default behavior may be more ideal (and I don't think world_rank is the right thing here, based on the documentation - should be local_rank if anything)
                                 shuffle=current_model_attrs.is_train)

    current_model_dataloader = DataLoader(current_model_attrs.dataset, 
                                          batch_size=current_model_attrs.BATCH_SIZE,
                                          sampler=sampler,
                                          shuffle=False, 
                                          num_workers=num_workers, 
                                          pin_memory=True, 
                                          persistent_workers=True)
    
    if rank==0:
        with open(TRAINING_LOG_FILEPATH, "a") as file:
            now = dt.datetime.now()
            current_time = now.strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"Data loaded | Current time = {current_time} \n")
    
    
    # Optimizer and Loss
    optimizer = torch.optim.AdamW(current_model_attrs.model.parameters(), lr=INITIAL_LEARNING_RATE, betas=[0.5,0.999])
    loss_function = torch.nn.L1Loss()
    
    
    current_model_attrs.model.train()
    
    lowest_loss = 999999999
    
    for epoch in range(1,upper_bound):
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        start = time.time()
        for i, (inputs,labels) in enumerate(current_model_dataloader):    
            start_batch = time.time()            
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
    
            optimizer.zero_grad()
    
            outputs = current_model_attrs.model(inputs)#.float()) #weird datatype mismatching... for some reason it's seeing HRRR data as double
            loss = loss_function(outputs,labels)
            
            loss.backward()
            optimizer.step()
    
            epoch_loss += loss.item()

            divisor = (int(len(current_model_dataloader)/100))
            if rank==0:
                if divisor: #(i.e. divisor > 0, when len(dataloader) > 100)
                    if (i+1)%divisor==0:
                        print(f"[Rank {rank}] Done with batch {i+1}/{len(current_model_dataloader)}. Time = {time.time()-start_batch:.2f} sec. Running loss = {epoch_loss/(i+1):.5f}")
                else:
                    if (i+1)%(int(len(current_model_dataloader)/10))==0:
                        print(f"[Rank {rank}] Done with batch {i+1}/{len(current_model_dataloader)}. Time = {time.time()-start_batch:.2f} sec. Running loss = {epoch_loss/(i+1):.5f}")
            
        if rank==0:
            print(f"End of epoch {epoch} | Average loss for epoch = {epoch_loss/len(current_model_dataloader):.5f} | Time for epoch = {time.time()-start:.1f} sec")

        was_model_saved=False
        if epoch_loss <= lowest_loss: #only save models that have lower loss than the previous best
            lowest_loss = epoch_loss
            #Saving with .module.state_dict() should suffice for DDP as well
            if rank==0: #don't want it to save many times at once, if loss is low across nodes
                torch.save(current_model_attrs.model.module.state_dict(), f"{TRAINED_MODEL_SAVEPATH}/{current_model_attrs.savename}{appendation}_TEMP.pt")
                print(f"New lowest loss - model saved @ epoch {epoch}")
            was_model_saved=True


        model_saved_str=""
        if was_model_saved:
            model_saved_str=f"| MODEL SAVED"
        
        if rank==0:
            with open(TRAINING_LOG_FILEPATH, "a") as file: 
                file.write(f"End of epoch {epoch} | Average loss for epoch = {epoch_loss/len(current_model_dataloader):.5f} | Time for epoch = {time.time()-start:.1f} sec {model_saved_str} \n")
            
    if rank==0:
        print(f"TRAINING FINISHED")
        with open(TRAINING_LOG_FILEPATH, "a") as file: 
            now = dt.datetime.now()
            current_time = now.strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"Training finished | Current time = {current_time} \n")
        
        os.rename(f"{TRAINED_MODEL_SAVEPATH}/{current_model_attrs.savename}{appendation}_TEMP.pt", 
                  f"{TRAINED_MODEL_SAVEPATH}/{current_model_attrs.savename}{appendation}.pt") #so if training is interrupted, previously saved model under the same name isn't wiped out
    
    dist.destroy_process_group()

    #return