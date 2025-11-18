
from FunctionsAndClasses.HEADER_torch import *
from FunctionsAndClasses.HEADER_utilities import *
from FunctionsAndClasses.CONSTANTS import *
from FunctionsAndClasses.utils import *

C = CONSTANTS()

######################################################################################################################################################

class HRRR_URMA_Dataset_AllVars_CONUS(Dataset):
    def __init__(self, 
                 is_train=True,
                 is_patches=True, #default to training setup, which is with patches
                 predictor_vars = ["pressurf", "t2m", "d2m", "spfh2m", "u10m", "v10m"],
                 target_vars = ["pressurf", "t2m", "d2m", "spfh2m", "u10m", "v10m"],
                 with_terrains=["hrrr","urma","diff"]
                ):
        
        """
        OPTIONS:
        ---------------------------------------------------------------
        - is_train --> bool to load either training or testing datasets
        - predictor_vars --> list of strings of the predictor variables to use, selected by their variable name. Valid options as of 2025/09/23: 
            - "pressurf"
            - "t2m"
            - "d2m"
            - "spfh2m"
            - "u10m"
            - "v10m"
        - target_vars --> list of strings of the target variables to use, selected by their variable name. Valid options = same as predictor_vars
        - with_terrains --> list of terrains to include as separate, normalized channels (can be None to not include terrain):
            - "hrrr" --> include NORMALIZED 2.5 km downscaled HRRR terrain field as a separate channel
            - "urma" --> include NORMALIZED 2.5 km nam_smarttopconus2p5 terrain field as a separate channel
            - "diff" --> include NORMALIZED map of the difference between HRRR and URMA terrain as a separate channel
            - !!!! NOTE: if both "hrrr" and "urma" are included, then HRRR terrain field will be normalized with respect to the mean/stddev of the URMA terrain! According to Ryan Lagerquist, best to use one norm for both terrains in this case


        DEPRECATED OPTIONS (should NOT be changed from default values)
        ---------------------------------------------------------------
        - REMOVED FROM THIS DATASET - see previous datasets, if those options are needed in the future. This class is only intended to load all times, with 1-hour HRRR forecasts as the predictor data, so the deprecated options are useless
        """

        #########################################
        ## Initialize vars

        self.C = CONSTANTS()
        
        self.is_train = is_train
        if self.is_train:
            print(f"is_train = {self.is_train} (2021/22/23)")
        else:
            print(f"is_train = {self.is_train} (2024)")
        
        self.is_patches = is_patches
        
        self.predictor_vars = predictor_vars
        self.target_vars = target_vars
        
        self.with_terrains = with_terrains
        self.with_hrrr_terrain = False
        self.with_urma_terrain = False
        self.with_diff_terrain = False
        if self.with_terrains is not None:
            if "hrrr".casefold() in [x.casefold() for x in self.with_terrains]: #more complex check to allow for whatever casing, as all lowercase might not necessarily be the best
                self.with_hrrr_terrain = True
            if "urma".casefold() in [x.casefold() for x in self.with_terrains]:
                self.with_urma_terrain = True
            if "diff".casefold() in [x.casefold() for x in self.with_terrains]:
                self.with_diff_terrain = True

        #This is static for a given patch size; preload it so it doesn't get called every time in __getitem__ (though the rest of it does need to be called there)
        self.valid_region_for_patch_sw_corner = get_valid_region_mask(PATCH_SIZE=C.PATCH_SIZE) #might need to be tweaked

        ######################################
        ## Initialize arrays to contain each variable's data/attributes
        
        self.xr_datasets_pred = [] #list of length == len(self.predictor_vars), containing the raw xarray dataset for each predictor variable, in order
        self.xr_datasets_targ = [] #same, but for target vars
        

        #These will be a list of lists; these sublists will be of length 1 
        self.datasets_pred_normed_means = [] 
        self.datasets_targ_normed_means = []
        self.datasets_pred_normed_stddevs = [] 
        self.datasets_targ_normed_stddevs = []

        
        #########################################
        ## Normalize terrain field

        terrain_path_hrrr, terrain_path_urma = self.get_var_filepath("terrain")
        if self.with_hrrr_terrain:
            self.xr_terrain_hrrr = xr.open_dataarray(terrain_path_hrrr, decode_timedelta=True, engine='cfgrib')
        if self.with_urma_terrain:
            self.xr_terrain_urma = xr.open_dataarray(terrain_path_urma, decode_timedelta=True, engine='cfgrib')
        if self.with_diff_terrain: #needed in case one or both of HRRR or/and URMA terrain aren't included, but their difference is
            self.xr_terrain_hrrr = xr.open_dataarray(terrain_path_hrrr, decode_timedelta=True, engine='cfgrib')
            self.xr_terrain_urma = xr.open_dataarray(terrain_path_urma, decode_timedelta=True, engine='cfgrib')
                
        self.normalize_terrain()

        #########################################
        ## Load predictor and target datasets
        # Unlike previous dataloader, data is NOT loaded into memory, as it takes too long for the larger western domain and doing so doesn't speed up training
        # Also unlike previous, normalization of the data does NOT happen here, but rather in __getitem__
        # Might want to explore ways of caching the data, but doesn't seem to be needed for now

        if is_train:
            train_test_str = 'train'
        else:
            train_test_str = 'test'
        
        for var_name in self.predictor_vars:
            start = time.time()
            data_save_path = self.get_var_filepath(var_name, is_pred=True)
            
            self.xr_datasets_pred.append(xr.open_dataarray(data_save_path, decode_timedelta=True, engine='cfgrib'))
            
            self.datasets_pred_normed_means.append(self.C.hrrr_means_dict[train_test_str][var_name])
            self.datasets_pred_normed_stddevs.append(self.C.hrrr_stddevs_dict[train_test_str][var_name])
            print(f"Predictor data for {var_name} loaded. Time taken = {time.time()-start:.1f} sec")

        for var_name in self.target_vars:
            start = time.time()
            data_save_path = self.get_var_filepath(var_name, is_pred=False)

            self.xr_datasets_targ.append(xr.open_dataarray(data_save_path, decode_timedelta=True, engine='cfgrib'))
            
            self.datasets_targ_normed_means.append(self.C.urma_means_dict[train_test_str][var_name]) 
            self.datasets_targ_normed_stddevs.append(self.C.urma_stddevs_dict[train_test_str][var_name])
            print(f"Target data for {var_name} loaded. Time taken = {time.time()-start:.1f} sec")
            
        self.predictor_indices = np.arange(len(self.xr_datasets_pred[0]))
        self.target_indices = np.arange(len(self.xr_datasets_targ[0]))

        assert len(self.predictor_indices) == len(self.target_indices), "Predictor indices array should be of the same length as the target indices array"

        print("DATASET CONSTRUCTION DONE")
        
    ######################################### FUNCTIONS #########################################
    
    def __len__(self):
        return len(self.predictor_indices)

    #########################################
    
    def normalize_terrain(self): #Put into a separate method just to clean up main
        if self.with_hrrr_terrain and not self.with_urma_terrain:
            terrain_hrrr = self.xr_terrain_hrrr.data
            self.terrain_hrrr_mean = np.nanmean(terrain_hrrr)
            self.terrain_hrrr_std = np.nanstd(terrain_hrrr)
            self.terrain_hrrr_normed = (terrain_hrrr - self.terrain_hrrr_mean)/self.terrain_hrrr_std
        if self.with_urma_terrain:
            terrain_urma = self.xr_terrain_urma.data
            self.terrain_urma_mean = np.nanmean(terrain_urma)
            self.terrain_urma_std = np.nanstd(terrain_urma)
            self.terrain_urma_normed = (terrain_urma - self.terrain_urma_mean)/self.terrain_urma_std
        if self.with_diff_terrain:
            terrain_hrrr = self.xr_terrain_hrrr.data
            terrain_urma = self.xr_terrain_urma.data
            terrain_diff = terrain_hrrr-terrain_urma
            self.terrain_diff_mean = np.nanmean(terrain_diff)
            self.terrain_diff_std = np.nanstd(terrain_diff)
            self.terrain_diff_normed = (terrain_diff - self.terrain_diff_mean)/self.terrain_diff_std
        if self.with_hrrr_terrain and self.with_urma_terrain: #use the same mean/std to norm both. Using URMA at the moment
            # Note in this case the URMA terrain stuff has already been done
            terrain_hrrr = self.xr_terrain_hrrr.data
            self.terrain_hrrr_mean = np.nanmean(terrain_hrrr) #for diagnostics only
            self.terrain_hrrr_std = np.nanstd(terrain_hrrr) #for diagnostics only
            self.terrain_hrrr_normed = (terrain_hrrr - self.terrain_urma_mean)/self.terrain_urma_std
        if self.with_terrains is not None:
            print(f"Terrain normalization done for {self.with_terrains}") 
        return

    #########################################

    def get_var_filepath(self, var_name, is_pred=True):
        """
        Inputs: 
            - var_name as string (e.g. "t2m", "d2m", etc). Can be "terrain" to load terrain data as well
            - is_pred = bool to control if predictor or target filepath is returned
        Output: relevant filepath(s) for that var
        """
        
        if var_name == "terrain":
            terrain_path_hrrr = f"{self.C.DIR_TRAIN_TEST}/terrain_CONUS_HRRR_2p5km.grib2"
            terrain_path_urma = f"{self.C.DIR_TRAIN_TEST}/terrain_CONUS_URMA_2p5km.grib2"
            return terrain_path_hrrr, terrain_path_urma
        else:
            if self.is_train:
                if is_pred:
                    data_save_path = f"{self.C.DIR_TRAIN_TEST}/train_hrrr_alltimes_CONUS_{var_name}_f01.grib2" #{str(self.forecast_lead_time).zfill(2)}.grib2"
                else:
                    data_save_path = f"{self.C.DIR_TRAIN_TEST}/train_urma_alltimes_CONUS_{var_name}.grib2"
            else:
                if is_pred:
                    data_save_path = f"{self.C.DIR_TRAIN_TEST}/test_hrrr_alltimes_CONUS_{var_name}_f01.grib2" #{str(self.forecast_lead_time).zfill(2)}.grib2"
                else:
                    data_save_path = f"{self.C.DIR_TRAIN_TEST}/test_urma_alltimes_CONUS_{var_name}.grib2"
            
            return data_save_path

    #########################################

    def get_normed_data_at_idx(self, i, idx, is_pred=True):
        """
        Helper function to be used in __getitem__. 
        Note: relies on datasets/means/stddevs in list to be ordered the same as the order in predictor_vars or target_vars, but the lists are constructed this way in the main function, so not that big of a deal for use in __getitem__. Be careful if calling this in an outside script, though!
        Inputs:
            - i = index of current variable in relation to predictor_vars or target_vars
            - idx = actual index to select
            - is_pred = bool to select from the correct list

        Outputs:
            - output of variable @ i, index=idx, normed appropriately, and appended with newaxis for concat purposes
        """
        if is_pred: #select from pred data
            return ((self.xr_datasets_pred[i][idx].data - self.datasets_pred_normed_means[i])/self.datasets_pred_normed_stddevs[i])[np.newaxis,:,:]
        else: #select from targ data
            return ((self.xr_datasets_targ[i][idx].data - self.datasets_targ_normed_means[i])/self.datasets_targ_normed_stddevs[i])[np.newaxis,:,:]

    #########################################

    def get_normed_data_at_idx_and_patch(self, i, idx, coords, is_pred=True):
        """
        Helper function to be used in __getitem__. 
        Note: relies on datasets/means/stddevs in list to be ordered the same as the order in predictor_vars or target_vars, but the lists are constructed this way in the main function, so not that big of a deal for use in __getitem__. Be careful if calling this in an outside script, though!
        Inputs:
            - i = index of current variable in relation to predictor_vars or target_vars
            - idx = actual time index to select
            - coords = 4-tuple array containing the indices of the domain to be selected, already calculated in the calling function.
                - ORDERING: ['south_lat_idx', 'north_lat_idx', 'west_lon_idx', 'east_lon_idx']
                - Should be the case that north_lat_idx = south_lat_idx+C.PATCH_SIZE, and likewise east_lon_idx = west_lon_idx+C.PATCH_SIZE, but this is left flexible, just in case
            - is_pred = bool to select HRRR or URMA (True/False respectively)

        Outputs:
            - output of variable @ i, patch=patch_idx, index=idx, normed appropriately, and appended with newaxis for concat purposes
        """
        
        if is_pred: #select from pred data
            return ((self.xr_datasets_pred[i][idx][coords[0]:coords[1], coords[2]:coords[3]].data - self.datasets_pred_normed_means[i])/self.datasets_pred_normed_stddevs[i])[np.newaxis,:,:]
        else: #select from targ data
            return ((self.xr_datasets_targ[i][idx][coords[0]:coords[1], coords[2]:coords[3]].data - self.datasets_targ_normed_means[i])/self.datasets_targ_normed_stddevs[i])[np.newaxis,:,:]

    #########################################
    

    def __getitem__(self, idx):
        
        if self.is_patches:
            # This block is much more verbose than it needs to be, but it helps for debugging
            list_of_valid_indices = np.argwhere(self.valid_region_for_patch_sw_corner==True) #convenient to select the SW corner from
            int_rand = np.random.randint(0, len(list_of_valid_indices))
            sw_corner_idxs = list_of_valid_indices[int_rand] #returns 2-tuple of SW corner [lat_idx, lon_idx]
    
            # ORDERING: ['south_lat_idx', 'north_lat_idx', 'west_lon_idx', 'east_lon_idx']
            # Useful to have this as an object variable rather than local, for plotting purposes in outside functions
            self.coords = [sw_corner_idxs[0], sw_corner_idxs[0]+C.PATCH_SIZE,
                           sw_corner_idxs[1], sw_corner_idxs[1]+C.PATCH_SIZE]
    
            ## Start with the first variable for each of predictor and target
            predictor = self.get_normed_data_at_idx_and_patch(0, idx, self.coords, is_pred=True)
            target = self.get_normed_data_at_idx_and_patch(0, idx, self.coords, is_pred=False)
    
            ## Add new channels for as many variables as we have
            if len(self.predictor_vars) > 1:
                for i, var_name in enumerate(self.predictor_vars[1:]): #don't double up on index 0
                    ds = self.get_normed_data_at_idx_and_patch(i, idx, self.coords, is_pred=True)
                    predictor = np.concatenate((predictor, ds), axis=0)
            if len(self.target_vars) > 1:
                for i, var_name in enumerate(self.target_vars[1:]): #don't double up on index 0
                    ds = self.get_normed_data_at_idx_and_patch(i, idx, self.coords, is_pred=False)
                    target = np.concatenate((target, ds), axis=0)
    
            ## Add terrain layers as last channels
            if self.with_hrrr_terrain:
                terr = (self.terrain_hrrr_normed[self.coords[0]:self.coords[1], self.coords[2]:self.coords[3]])[np.newaxis,:,:]
                predictor = np.concatenate((predictor, terr), axis=0)
            if self.with_urma_terrain:
                terr = (self.terrain_urma_normed[self.coords[0]:self.coords[1], self.coords[2]:self.coords[3]])[np.newaxis,:,:]
                predictor = np.concatenate((predictor, terr), axis=0)
            if self.with_diff_terrain:
                terr = (self.terrain_diff_normed[self.coords[0]:self.coords[1], self.coords[2]:self.coords[3]])[np.newaxis,:,:]
                predictor = np.concatenate((predictor, terr), axis=0)

        else: #load all of CONUS
            ## Start with the first variable for each of predictor and target
            predictor = self.get_normed_data_at_idx(0, idx, is_pred=True)
            target = self.get_normed_data_at_idx(0, idx, is_pred=False)
    
            ## Add new channels for as many variables as we have
            if len(self.predictor_vars) > 1:
                for i, var_name in enumerate(self.predictor_vars[1:]): #don't double up on index 0
                    ds = self.get_normed_data_at_idx(i, idx, is_pred=True)
                    predictor = np.concatenate((predictor, ds), axis=0)
            if len(self.target_vars) > 1:
                for i, var_name in enumerate(self.target_vars[1:]): #don't double up on index 0
                    ds = self.get_normed_data_at_idx(i, idx, is_pred=False)
                    target = np.concatenate((target, ds), axis=0)

            ## Add terrain layers as last channels
            if self.with_hrrr_terrain:
                terr = (self.terrain_hrrr_normed)[np.newaxis,:,:]
                predictor = np.concatenate((predictor, terr), axis=0)
            if self.with_urma_terrain:
                terr = (self.terrain_urma_normed)[np.newaxis,:,:]
                predictor = np.concatenate((predictor, terr), axis=0)
            if self.with_diff_terrain:
                terr = (self.terrain_diff_normed)[np.newaxis,:,:]
                predictor = np.concatenate((predictor, terr), axis=0)

        
        return (predictor), (target)
