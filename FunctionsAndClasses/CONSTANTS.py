import os

class CONSTANTS():
    def __init__(self):

        """
        Stores constants/variables that are frequently used across different files and classes.
        """
        
        self.varname_translation_dict = {"pressurf":"sp",
                                         "t2m":"t2m",
                                         "d2m":"d2m",
                                         "spfh2m":"sh2",
                                         "u10m":"u10",
                                         "v10m":"v10"}
        
        self.urma_var_select_dict = {"sp":{'filter_by_keys':{'typeOfLevel': 'surface'}},
                                     "t2m":{'filter_by_keys':{'typeOfLevel': 'heightAboveGround','level':2}}, 
                                     "d2m":{'filter_by_keys':{'typeOfLevel': 'heightAboveGround','level':2}}, 
                                     "sh2":{'filter_by_keys':{'typeOfLevel': 'heightAboveGround','level':2}},
                                     "u10":{'filter_by_keys':{'typeOfLevel': 'heightAboveGround','level':10}},
                                     "v10":{'filter_by_keys':{'typeOfLevel': 'heightAboveGround','level':10}}}

        # For plot labeling purposes
        self.varname_units_dict = {"pressurf":"Pa",
                                   "t2m":"deg K",
                                   "d2m":"deg K",
                                   "spfh2m":"kg/kg",
                                   "u10m":"m/s",
                                   "v10m":"m/s"}
        
        
        # Constants for (pseudo-)normalization
        self.hrrr_means_dict = {'train':{'pressurf':88264.4,
                                         't2m':284.451, 
                                         'd2m':273.703, 
                                         'spfh2m':0.00529336, 
                                         'u10m':1.17153, 
                                         'v10m':-0.313557}, 
                                'test':{'pressurf':88250.4,
                                         't2m':284.61, 
                                         'd2m':273.83, 
                                         'spfh2m':0.00530784, 
                                         'u10m':1.19730, 
                                         'v10m':-0.339266} }
        
        self.hrrr_stddevs_dict = {'train':{'pressurf':8493.89,
                                         't2m':11.0624, 
                                         'd2m':9.32916, 
                                         'spfh2m':0.00311317, 
                                         'u10m':3.14697, 
                                         'v10m':3.69828}, 
                                  'test':{'pressurf':8490.22,
                                         't2m':10.9822, 
                                         'd2m':9.14753, 
                                         'spfh2m':0.00306908, 
                                         'u10m':3.14104, 
                                         'v10m':3.71392} }
        
        self.urma_means_dict = {'train':{'pressurf':88264.4,
                                         't2m':284.451, 
                                         'd2m':273.703, 
                                         'spfh2m':0.00529336, 
                                         'u10m':1.17153, 
                                         'v10m':-0.313557}, 
                                'test':{'pressurf':88250.4,
                                         't2m':284.61, 
                                         'd2m':273.83, 
                                         'spfh2m':0.00530784, 
                                         'u10m':1.19730, 
                                         'v10m':-0.339266} }

        self.urma_stddevs_dict = {'train':{'pressurf':8493.89,
                                         't2m':11.0624, 
                                         'd2m':9.32916, 
                                         'spfh2m':0.00311317, 
                                         'u10m':3.14697, 
                                         'v10m':3.69828}, 
                                  'test':{'pressurf':8490.22,
                                         't2m':10.9822, 
                                         'd2m':9.14753, 
                                         'spfh2m':0.00306908, 
                                         'u10m':3.14104, 
                                         'v10m':3.71392} }
        
        

        ## Common directories 
        self.DIR_TRAIN_TEST = f"/scratch3/BMC/wrfruc/aschein/Train_Test_Files"
        self.DIR_UNET_MAIN = os.getcwd() #f"/scratch3/BMC/wrfruc/aschein/UNet_main"
        self.DIR_TRAINED_MODELS = f"{os.getcwd()}/Trained_models" #f"/scratch3/BMC/wrfruc/aschein/UNet_main/Trained_models"
        self.DIR_SMARTINIT_DATA = f"/scratch3/BMC/wrfruc/aschein/SMARTINIT_STUFF/smartinit_2024/output_files" #regridded Smartinit
        self.DIR_SMARTINIT_DATA_NDFD_GRID = f"/scratch3/BMC/wrfruc/aschein/SMARTINIT_STUFF/smartinit_2024/output_files_NDFDgrid" #old Smartinit on native grid

        ## Indexes for selection of patches for training
        self.PATCH_SIZE=160 #400

        