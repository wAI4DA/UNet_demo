
from FunctionsAndClasses.HEADER_utilities import *
from FunctionsAndClasses.HEADER_torch import *
from FunctionsAndClasses.HEADER_HRRR_URMA_Datasets_AllVars import *
from FunctionsAndClasses.HEADER_models import *
from FunctionsAndClasses.CONSTANTS import *
from FunctionsAndClasses.utils import*

C = CONSTANTS()

######################################################################################################################################################

class DefineModelAttributes():
    """
    Class to aid in constructing models and their Pytorch datasets for use in training. Must set an object of this class whose attributes can then be used to initialized and control model training.
    """
    def __init__(self,
                 is_train=True,
                 is_patches=False,
                 is_western_domain=False,
                 is_attention_model=False,
                 is_residual_model=False,
                 BATCH_SIZE=1,
                 NUM_EPOCHS=1,
                 predictor_vars=["pressurf", "t2m", "d2m", "spfh2m", "u10m", "v10m"],
                 target_vars=["pressurf", "t2m", "d2m", "spfh2m", "u10m", "v10m"],
                 with_terrains=["diff"],
                 ### Below here = deprecated/obsolete; should not be changed from default values!
                 months=[1,12],  
                 hours="all", 
                 forecast_lead_time=1
                ):
            
        """
        !!!! See HRRR_URMA_Datasets_AllVars_... class(es) for any variable definitions/restrictions not listed below !!!! 
        Variables unique to DefineModelAttributes:
            - is_patches --> (added 2025-09-10) bool to control if the .dataset will be from HRRR_URMA_Datasets_AllVars_Patches (if True) or HRR_URMA_Datasets_AllVars_Western_Domain (if false). Should be True for training and False for testing, generally
            - is_western_domain --> (added 2025-10-13) bool to control if the selected domain is the western domain or the entirety of CONUS
            - is_attention_model --> sets savename to include "attn" at the start, and should be used for model selection in calling function
            - is_residual model --> sets savename to include "residual" at the start, and should be used for model selection in calling function
            
        """
    
        #########################################

        self.C = CONSTANTS()
        
        self.is_train = is_train
        self.is_patches = is_patches
        self.is_western_domain = is_western_domain
        self.is_attention_model = is_attention_model
        self.is_residual_model = is_residual_model
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_EPOCHS = NUM_EPOCHS
        self.predictor_vars = predictor_vars
        self.target_vars = target_vars
        self.with_terrains = with_terrains
        
        self.months = months
        self.hours = hours
        self.forecast_lead_time = forecast_lead_time
        

        self.create_save_name()

        self.dataset = None #call create_dataset() in whatever calling function needs it
        self.num_channels_in = None
        self.num_channels_out = None

        self.model = None #call set_model_architecture() in whatever calling function needs it
        
    ######################################### FUNCTIONS #########################################
    
    def create_save_name(self):
        """
        Savename is additive, i.e. only includes attributes present in the model
        Ordering: 
            - "residual" if is_residual_model
            - "attn" if is_attention_model
            - "CONUS" if not is_western_domain
            - Batch size (BS#)
            - Num epochs (NE#)
            - Terrains (tH, tU, tD)
            - Predictor variables, in parentheses, separated by dashes
                Example: pred(t2m-d2m-u10m)
            - Target variables, in parentheses, separated by dashes
                Example: targ(t2m-pressurf)

            !! OBSOLETE - should not be used - included as a legacy option
            - Months IF NOT 1-12 
            - Hours IF NOT all 
            - Forecast lead time IF NOT 1 
            
        """
        ## Define all optional/constructive arguments
        
        attn_str = ""
        residual_str = ""
        conus_str = ""
        terrain_str = ""
        
        if self.is_attention_model:
            attn_str = "attn"
        if self.is_residual_model:
            residual_str = "residual"
        if not self.is_western_domain:
            conus_str = "CONUS"
        terrain_str = "_".join([f"t{x[0].capitalize()}" for x in self.with_terrains]) if self.with_terrains is not None else ""


        ### DEPRECATED/OBSOLETE - should not be used, and should not appear in savename, but here for legacy. 
        # Note that set_model_attrs_from_savename will fail to recognize these attributes!
        month_str = ""
        hours_str = ""
        forecast_str = ""
        if self.months != [1,12]:
            month_str = f"months{self.months[0]}-{self.months[1]}"
        if self.hours != "all":
            hours_str = f"hours{'-'.join([str(hour) for hour in self.hours])}"
        if self.forecast_lead_time != 1:
            forecast_str = f"f{str(forecast_lead_time).zfill(2)}"
        ### End deprecated/obsolete block

        optional_str_list = [terrain_str, month_str, hours_str, forecast_str]
        optional_str = "_".join([x for x in optional_str_list if x != ""])
        
        # Doesn't play nice if defined within the f-string
        # Making as variables so these can be called independently for other plotting purposes (don't forget to add "pred" and "targ" in calling code!)
        self.pred_str = "-".join(self.predictor_vars)
        self.targ_str = "-".join(self.target_vars)

        savename_attrs_list = [residual_str, 
                               attn_str,
                               conus_str,
                               f"BS{self.BATCH_SIZE}", 
                               f"NE{self.NUM_EPOCHS}", 
                               f"{optional_str}", 
                               f"pred({self.pred_str})", 
                               f"targ({self.targ_str})"]
        
        self.savename = "_".join([x for x in savename_attrs_list if x !=""])
        
        return 

    #########################################

    def create_dataset(self):
        """ 
        Creates the requisite Dataset for use in Pytorch Dataloader
        NOT called by default (due to calculation expense) - must be invoked by calling function 
        """

        # Check if this was already done, to save duplicate computation
        if self.dataset is None:
            print(f"Making dataset for model {self.savename}")

            if self.is_western_domain:
                print(f"Making western domain dataset")
                if self.is_patches:
                    print(f"is_patches = {self.is_patches}; making patches dataset")
                    self.dataset = HRRR_URMA_Dataset_AllVars_Patches(is_train = self.is_train,
                                                                     with_terrains = self.with_terrains, 
                                                                     predictor_vars = self.predictor_vars,
                                                                     target_vars = self.target_vars)
                else:
                    print(f"is_patches = {self.is_patches}; making full domain dataset") 
                    self.dataset = HRRR_URMA_Dataset_AllVars_Western_Domain(is_train = self.is_train,
                                                                             with_terrains = self.with_terrains, 
                                                                             predictor_vars = self.predictor_vars,
                                                                             target_vars = self.target_vars)
            else: #Make CONUS dataset
                print(f"Making CONUS dataset")
                if self.is_patches:
                    print(f"is_patches = {self.is_patches}; making patches dataset")
                else:
                    print(f"is_patches = {self.is_patches}; making full domain dataset")
                self.dataset = HRRR_URMA_Dataset_AllVars_CONUS(is_train = self.is_train,
                                                               is_patches = self.is_patches,
                                                               with_terrains = self.with_terrains,
                                                               predictor_vars = self.predictor_vars,
                                                               target_vars = self.target_vars)
                


                
            # Returns a list of dt.datetime objects of all dates in the current dataset. Useful for plotting and time series stuff
            self.dataset_date_list = [dt.datetime.strptime(str(np.datetime_as_string(date, unit='s')), "%Y-%m-%dT%H:%M:%S") for date in self.dataset.xr_datasets_pred[0].valid_time.data]
        
        else:
            print(f"Dataset for the model {self.savename} was already computed. If it needs to be recomputed, set [current model].dataset=None and rerun .create_dataset()")
        
        self.num_channels_in = np.shape(self.dataset[0][0])[0]
        self.num_channels_out = np.shape(self.dataset[0][1])[0] #Don't forget to put this in the model definition when targeting >1 var!!!
        return

    #########################################
    
    def set_model_attrs_from_savename(self, savename):
        """
        Input: str of model savename, formatted as in self.create_save_name() - does NOT need a filepath, only the savename
        Sets the following model attributes:
            - is_attention_model
            - is_residual_model
            - BATCH_SIZE
            - NUM_EPOCHS
            - with_terrains (the whole list)
            - predictor_vars (the whole list)
            - target_vars (the whole list)
        """
        self.with_terrains = []
        strs = savename.split("_")
        for string in strs:
            #(7/10) as currently written, will fail to detect some attrs for models with "months", "hours" in the name, but I'm not planning on dealing with such models for now, so fix this if needed
            if "attn" in string:
                self.is_attention_model=True
            elif "residual" in string:
                self.is_residual_model=True
            # elif "CONUS" in string: ### NOT including this - it's perfectly valid to load a CONUS-trained model but have a western domain dataset for testing
            #     self.is_western_domain=False
            elif "BS" in string:
                self.BATCH_SIZE = int("".join([char for char in string if char.isdigit()]))
            elif "NE" in string:
                self.NUM_EPOCHS = int("".join([char for char in string if char.isdigit()]))
            elif string=="tH":
                self.with_terrains.append("hrrr")
            elif string=="tU":
                self.with_terrains.append("urma")
            elif string=="tD":
                self.with_terrains.append("diff")
            elif "pred(" in string:
                self.predictor_vars = ((string.split("(")[1])[:-1]).split("-")
            elif "targ(" in string:
                if ".pt" in string: #bad hack but w/e... this whole function is a bad hack
                    self.target_vars = ((string.split("(")[1])[:-4]).split("-")
                else:
                    self.target_vars = ((string.split("(")[1])[:-1]).split("-")
        
        print(f"Model attributes set from {savename}")
        self.create_save_name() #needs to be re-called to properly set attributes
        
        if "CONUS" in self.savename and "CONUS" not in savename: #weird edge case where we want to load a western-domain-trained model (without "CONUS" in the original savename) for use over the whole CONUS domain, which in the current (2025-10-27) setup results in self.savename containing "CONUS" when it shouldn't. This is a very hacky "solution" but it will eventually go away as models are trained over CONUS and thus have that in their savename
            self.savename = self.savename.replace("_CONUS", "")
        
        print(f"Savename set to {self.savename}")
        return

    #########################################

    def set_model_architecture(self):
        """
        Sets model architecture based on parameters.
        Actually makes the model, which is attached to the object .
        This DOES NOT LOAD WEIGHTS!!!! The specific model to use is still the responsibility of the calling function! This is just to get the right framework
        """
        
        if self.dataset is None:
            print(f"Must run .create_dataset() before defining model architecture")
        else:
            if self.model is None:
                if self.is_attention_model and not self.is_residual_model:
                    self.model = UNet_Attention_simple(n_channels_in=self.num_channels_in, n_channels_out=self.num_channels_out)
                    print(f"Model architecture set: UNet_Attention_simple")
                elif self.is_residual_model and not self.is_attention_model:
                    self.model = UNet_Residual(n_channels_in=self.num_channels_in, n_channels_out=self.num_channels_out)
                    print(f"Model architecture set: UNet_Residual")
                elif self.is_residual_model and self.is_attention_model:
                    self.model = UNet_Residual_Attention(n_channels_in=self.num_channels_in, n_channels_out=self.num_channels_out)
                    print(f"Model architecture set: UNet_Residual_Attention")
                else: #default to simple UNet
                    self.model = UNet_simple(n_channels_in=self.num_channels_in, n_channels_out=self.num_channels_out)
                    print(f"Model architecture set: UNet_simple")
            else:
                print(f"Model architecture was already set previously! Set ''self.model=None'' and rerun this function if a new model architecture is desired")

        return

    #########################################

    def set_model_weights(self, model_savename, is_different_path=False):
        """
            Input = model savename to load 
                - Assumption is the model name does NOT INCLUDE THE PATH! (Assumed to be Trained_models) (Also NOT including ".pt" at the end!)
                - However, 'is_different_path' can be set to True, and then 'model_savename' should include the path, AND .pt AT THE END !!!
            This will first set the model attributes from the savename (unless is_different_path = True), then create the dataset if not done already, then load the model weights. The intention is for this to be the only necessary function to call after instantiation in order to get a functioning .model for that object

            Sets the model weights from saved model, then updates savename to match what's on disk, if they don't already match
            
            It's the calling function's responsibility to make sure the architecture matches the requested weights!
        """
        if not is_different_path:
            self.set_model_attrs_from_savename(model_savename)
        
        if self.dataset is None:
            self.create_dataset()
        
        if self.model is None:
            self.set_model_architecture() #if this should be skipped (e.g. doing a model with an architecture not supported in set_model_architecture), then make sure self.model is manually set in the calling function BEFORE using this one! 
        
        device = torch.device("cuda")
        self.model.to(device)
        if is_different_path:
            self.model.load_state_dict(torch.load(f"{model_savename}", weights_only=True))
        else:
            self.model.load_state_dict(torch.load(f"{self.C.DIR_TRAINED_MODELS}/{model_savename}.pt", weights_only=True))

        print(f"Weights for {model_savename} loaded")

        if (not is_different_path) and (self.savename != model_savename):
            self.savename = model_savename #generally we want this to match the calling savename
            print(f"Savename set to ''{self.savename}'' - rerun .create_save_name() if this is not desired")
        
        return