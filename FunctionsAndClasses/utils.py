
from FunctionsAndClasses.HEADER_torch import *
from FunctionsAndClasses.HEADER_utilities import *
from FunctionsAndClasses.HEADER_plotting import *
from FunctionsAndClasses.HEADER_HRRR_URMA_Datasets_AllVars import *
from FunctionsAndClasses.HEADER_models import *
from FunctionsAndClasses.DefineModelAttributes import *
from FunctionsAndClasses.CONSTANTS import *


######################################################################################################################################################

def get_model_output_at_idx(model_attrs, 
                            model, 
                            pred_var="t2m", 
                            targ_var="t2m", 
                            idx=0, 
                            is_nan=False,
                            nan_fill_value=0,
                            is_unnormed=True,
                            crop_pred=False,
                            crop_model_output=False,
                            crop_targ=False,
                            device="cuda"
                           ):
    """
    Inputs:
        - model_attrs: DefineModelAttributes object. MUST HAVE .create_dataset() ALREADY CALLED! 
        - model: Pytorch model to use, with weights loaded and device initialized
        - pred_var: string for the predictor variable to get the output of
        - targ_var: string for the target variable to get the output of
        - idx: index to get the output of (time index)
        - is_nan: bool, should generally be set to True if predictor data has NaNs (e.g. CONUS HRRR data) because the models don't apply to NaNs and thus the output is severely truncated from what it should be. 
            > Leaving control to the calling function as of 2025-10-15, but might update this to be automatic if any NaN is detected in the predictor
        - nan_fill_value: int or float, used to fill in all NaN values in the predictor data, if is_nan=True
        - is_unnormed: bool; if True (default), returns unnormed data. Predictor and target data is read directly from their raw xarray files, whilst model_output is unnormalized by the corresponding variable's stored (pseudo-) mean and stddev
        - crop_pred: bool, default value = False; if True, crops the predictor data to cut out regions of NaNs for better plotting. This also serves as the mask for crop_model_output and crop_targ
        - crop_model_output: bool, default value = False; if True (default), crops the model output data according to the predictor var's mask
        - crop_targ: bool, default value = False; if True (default), crops the target data " "
        - device: cuda device, default to just "cuda". Might need to change this in calling function, be careful

    Outputs:
        - predictor @ index, UNNORMED if is_unnormed, CROPPED if crop_pred
        - target @ index, UNNORMED if is_unnormed, CROPPED if crop_model_output
        - model output @ index, UNNORMED if is_unnormed, CROPPED if crop_targ
        - dt_current as dt.datetime object, for plot title purposes
    """
    
    pred,targ = model_attrs.dataset[idx]
    if is_nan: #Added 2025-10-15
        np.nan_to_num(pred, copy=False, nan=nan_fill_value) 
    pred = pred[np.newaxis,:] 
    pred_gpu = torch.from_numpy(pred).cuda(device)
    
    with torch.no_grad():
        model_output = model(pred_gpu.float())
        model_output = model_output.cpu().numpy()
    
    date = model_attrs.dataset.xr_datasets_pred[model_attrs.predictor_vars.index(pred_var)][idx].valid_time.data
    dt_current = dt.datetime.strptime(str(np.datetime_as_string(date, unit='m')), "%Y-%m-%dT%H:%M")
    
    if is_unnormed:
        pred = model_attrs.dataset.xr_datasets_pred[model_attrs.predictor_vars.index(pred_var)][idx].data
        targ = model_attrs.dataset.xr_datasets_targ[model_attrs.target_vars.index(targ_var)][idx].data

        model_output = ( model_attrs.dataset.datasets_targ_normed_stddevs[model_attrs.target_vars.index(targ_var)]
                         *model_output[0,model_attrs.target_vars.index(targ_var),:] 
                         + model_attrs.dataset.datasets_targ_normed_means[model_attrs.target_vars.index(targ_var)] )
    
    else: #model output is already normed
        pred = pred[0,model_attrs.predictor_vars.index(pred_var),:]
        targ = targ[model_attrs.target_vars.index(targ_var),:]

    if crop_model_output:
        model_output = crop_input(model_output, pred)
    if crop_targ:
        targ = crop_input(targ, pred)
    if crop_pred: #pred done last because it first has to serve as the mask for the previous data
        pred = crop_input(pred, pred)
    
    return pred, targ, model_output, dt_current


########################################################

def crop_input(data_to_crop, data_for_mask):
    """
    Crops out areas of only NaNs. Note this takes 2 input arguments, which can be the same but usually aren't for cropping over CONUS...
    Inputs:
    - data_to_crop = input 2D numpy array that we want cropped. Does NOT need to have regions of NaNs!
    - data_for_mask = input 2D numpy array that DOES have regions of NaNs, whose mask is to be used to crop data_to_crop

    Usual use case: data_for_mask = predictor data for HRRR, data_to_crop = whatever other data (could also be predictor data, or model output, or target data)
    
    """

    data_to_crop_xr = xr.DataArray(data_to_crop, dims=('y','x')) #Easiest to cast in xarray. (y,x) should be the ordering of the dims for such data
    data_for_mask_xr = xr.DataArray(data_for_mask, dims=('y','x')) #Also needs to be a DataArray for .where to work
    data_to_crop_xr_cropped = data_to_crop_xr.where(~np.isnan(data_for_mask_xr), drop=True)

    return data_to_crop_xr_cropped.data #return raw array, not xarray object


########################################################

def get_smartinit_output_at_idx(i, 
                                target_var,
                                FORECAST_LEAD_HOURS=1, 
                                smartinit_directory=None,
                                smartinit_var_select_dict=None, 
                                varname_translation_dict=None, 
                                crop_output=False,
                                IDX_MIN_LON=None,
                                IDX_MIN_LAT=None,
                                IMG_SIZE_LON=None,
                                IMG_SIZE_LAT=None,
                                START_DATE=None
                               ):
    """
    Method to open one Smartinit file and return its output for one variable, restricted to whatever spatial domain we define.
    Designed for StatObjectConstructor but can be called from anywhere else that Smartinit output is needed.

    Inputs:
        - i = int of index to select. Should line up with sample_idx indexing from HRRR
        - target_var = string of a valid target variable, e.g. "t2m"
        - FORECAST_LEAD_HOURS = int of forecast lead time. Default = 1. !!! Should already have offset START_DATE if START_DATE is not None !!!
        - smartinit_directory = directory of smartinit data which is NOT subset in any way but is named according to the convention in the code below. If None (default), autodirects to the smartinit directory in the CONSTANTS class.
        - smartinit_var_select_dict = as in StatObjectConstructor. If None (default), autodirects to the appropriate dict in the CONSTANTS class.
        - varname_translation_dict = as in StatObjectConstructor. If None (default), autodirects to the appropriate dict in the CONSTANTS class.
        - crop_output = bool to determine if the output is cropped to the target region (e.g. western domain). Default=False, so manually set this to True in the calling function if Smartinit over only a subset region is desired, otherwise the raw output over all of CONUS (including the NaN regions!) will be returned
        - IDXs and IMG_SIZEs = define region as usual. Should be the same as whatever model domain being compared against. 
            > Native Smartinit grid (=NDFD) is misaligned with the regridded HRRR/URMA grid!
            > Fixed by regridding the Smartinit data as of 2025/10/22, but this changes the indices previously used for Smartinit's western region
            > New Smartinit western region = use the same indices as HRRR/URMA! 
            > For CONUS, after using this method, be careful of the bounds - they're not the same as regridded HRRR, so the union of their masks should be used when comparing their data
        - START_DATE = dt.datetime object. Currently (as of 7/29) only have Smartinit data for 2024, so this should be 2023/12/31 23z or later. 
            > CURRENTLY SHOULD NOT BE CHANGED FROM THE DEFAULT! 
            > !!! VERY IMPORTANT: if not None, then the calling function should have START_DATE = dt.datetime([20240101_00z or greater])-dt.timedelta(hours=FORECAST_LEAD_HOURS) !!!

    Outputs:
        - xarray object of the smartinit, sliced down to the variable and domain of interest.
            - Returns xarray object, not just data, so calling function should invoke .data if that's what's desired
    """

    C = CONSTANTS()

    #These should generally NOT be changed in the function call
    if smartinit_directory is None:
        smartinit_directory = C.DIR_SMARTINIT_DATA
    if smartinit_var_select_dict is None:
        smartinit_var_select_dict = C.urma_var_select_dict #they share the same keys
    if varname_translation_dict is None:
        varname_translation_dict = C.varname_translation_dict
    
    
    if START_DATE is None: #This should be the default, as this method is written to count hours from the first available Smartinit date, which is (as of 2025-08-26) 2024-01-01 00z
        START_DATE = dt.datetime(2024,1,1,0)-dt.timedelta(hours=FORECAST_LEAD_HOURS)
    
    DATE_STR = dt.date.strftime(START_DATE + dt.timedelta(hours=i), "%Y%m%d")
    file_to_open = f"{smartinit_directory}/hrrr_smartinit_{DATE_STR}_t{str((START_DATE.hour+i)%24).zfill(2)}z_f{str(FORECAST_LEAD_HOURS).zfill(2)}_regridded.grib2" #changed 2025-10-22 to read the regridded CONUS smartinit
    xr_smartinit = xr.open_dataset(file_to_open,
                                   engine="cfgrib", 
                                   backend_kwargs=smartinit_var_select_dict[varname_translation_dict[target_var]],
                                   decode_timedelta=True)
    xr_smartinit = xr_smartinit[varname_translation_dict[target_var]]
    
    if crop_output:
        xr_smartinit = xr_smartinit.isel(y=slice(IDX_MIN_LAT, IDX_MIN_LAT+IMG_SIZE_LAT),
                                         x=slice(IDX_MIN_LON, IDX_MIN_LON+IMG_SIZE_LON))
    
    return xr_smartinit

########################################################

def plot_model_vs_model_error(model_1_output, model_2_output, pred, targ, date_str, error_units, avg_denom=10):
    
    #Restriction done automatically, no options here
    #Do first restriction to HRRR domain (using pred), then second restriction to model_2_output
        # This second iteration will do nothing if both models are ML, but will restrict to the intersection of HRRR and Smartinit if model_2_output is Smartinit data
    targ = crop_input(targ, pred)
    model_1_output = crop_input(model_1_output, pred)
    model_2_output = crop_input(model_2_output, pred)
    pred = crop_input(pred, pred)

    pred = crop_input(pred, model_2_output)
    model_1_output = crop_input(model_1_output, model_2_output)
    targ = crop_input(targ, model_2_output)
    model_2_output = crop_input(model_2_output, model_2_output)

    maxtemp = np.max([np.nanmax(model_1_output.squeeze()), np.nanmax(model_2_output.squeeze()), np.nanmax(targ.squeeze()), np.nanmax(pred.squeeze())])
    mintemp = np.min([np.nanmin(model_1_output.squeeze()), np.nanmin(model_2_output.squeeze()), np.nanmin(targ.squeeze()), np.nanmin(pred.squeeze())])
    avg = (maxtemp-mintemp)/avg_denom
    
    fig, axs = plt.subplots(1,1, figsize=(12,16))

    # Need to plot difference in ABSOLUTE errors, otherwise there's issues with negative regions
    pos = axs.imshow((np.abs(model_1_output.squeeze()-targ.squeeze()) - (np.abs(model_2_output.squeeze()-targ.squeeze()))), cmap="coolwarm", origin='lower', vmin = -1*avg, vmax = avg)
    axs.axis("off")
    cbar = fig.colorbar(pos, fraction=0.022, pad=0.01)
    cbar.set_label(f"Difference {error_units}")
    
    plt.title(f"Model 1 error minus Model 2 error, {date_str}")

    return


########################################################

def get_valid_region_mask(VALID_REGION_FILEPATH=None, 
                          PATCH_SIZE=None, 
                          file_save_dir=None, 
                          file_save_name=None):
    """
    Function to either fetch a precalculated mask of valid locations for patch's SW corner selection for use in the CONUS Dataset class, or calculate and return+save that data if it doesn't exist in file_save_dir.
    SAVED FILES SHOULD/WILL BE OF DIMENSION 1597x2345, i.e. the entire extended HRRR/URMA domain, because this is what the Dataset class(es) use and what __getitem__ will expect!

    Inputs:
        - VALID_REGION_FILEPATH = full filepath (including extension - should be csv) to the file containing the binary valid/invalid location data, which should be formatted according to this function's writing functionality. If it exists, it's read in and returned, otherwise it's calculated then returned + saved to disk
            > If None, defaults to {file_save_dir}/{file_save_name}
        - PATCH_SIZE - int for square patch size. If None (default), then the PATCH_SIZE in the CONSTANTS class is used. Should usually be included in the calling function's use, just in case
        - file_save_dir = full filepath to the save directory, if not using the default location
            > Default location = /scratch3/BMC/wrfruc/aschein/Train_Test_Files/
        - file_save_name = savename (including .csv extension) of the file, if not using the default construction
            > Default savename = "valid_region_for_patch_sw_corner_PATCHSIZE{PATCH_SIZE}.csv" - this only works for square patches, so revisit this convention if nonsquare patches are used!
    """

    C = CONSTANTS()
    
    if PATCH_SIZE is None:
        PATCH_SIZE = C.PATCH_SIZE 

    if file_save_name is None:
        file_save_name = f"valid_region_for_patch_sw_corner_PATCHSIZE{PATCH_SIZE}.csv"

    if file_save_dir is None:
        file_save_dir = C.DIR_TRAIN_TEST

    if VALID_REGION_FILEPATH is None:
        VALID_REGION_FILEPATH = f"{file_save_dir}/{file_save_name}"
    
    if os.path.exists(VALID_REGION_FILEPATH): #read in the data 
        valid_region_for_patch_sw_corner = np.genfromtxt(VALID_REGION_FILEPATH, delimiter=',')
    else: #need to calculate the region, and save to disk then return
        print(f"Valid patch selection region for PATCH_SIZE = {PATCH_SIZE} doesn't exist on disk (should be at {VALID_REGION_FILEPATH}). Computing this region...")
        xr_hrrr_conus = xr.open_dataarray(f"{C.DIR_TRAIN_TEST}/test_hrrr_alltimes_CONUS_t2m_f01.grib2", decode_timedelta=True, engine='cfgrib') #using t2m as the default - doesn't matter
        data_mask = xr_hrrr_conus[0].notnull() #Somewhat bugged if calling this in the below selection, so it needs to be separate
        xr_hrrr_conus_masked = xr_hrrr_conus[0].where(data_mask, drop=False) #don't drop! Need to keep the dimension
        valid_region_for_patch_sw_corner =  np.zeros((np.shape(xr_hrrr_conus_masked.data)))
        
        start = time.time()
        for lat_idx in xr_hrrr_conus_masked.y.data[:-PATCH_SIZE]: #Can't extend a patch beyond the domain!
            for lon_idx in xr_hrrr_conus_masked.x.data[:-PATCH_SIZE]: #Can't extend a patch beyond the domain!
                patch = xr_hrrr_conus_masked[lat_idx:lat_idx+PATCH_SIZE, lon_idx:lon_idx+PATCH_SIZE]
                if ~np.any(patch.isnull()): #if the patch is ok, flip the corresponding index in valid_region_for_patch_sw_corner
                    valid_region_for_patch_sw_corner[lat_idx, lon_idx]=1

        with open(VALID_REGION_FILEPATH, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(valid_region_for_patch_sw_corner) 
        
        print(f"Valid region computed, time taken = {time.time() - start:.1f} seconds. New file located at {VALID_REGION_FILEPATH} for future use")

    return valid_region_for_patch_sw_corner

########################################################

def plot_predictor_output_truth_error_CONUS(predictor, 
                                            model_output,
                                            target,
                                            include_predictor=False,
                                            include_model_output=False,
                                            include_target=False,
                                            use_hrrr_mask=True,
                                            use_smartinit_mask=False,
                                            date_str="DATE",
                                            title="MODEL NAME",
                                            error_units="",
                                            save_fig=False, 
                                            save_dir=f"/scratch3/BMC/wrfruc/aschein/UNet_main", 
                                            fig_savename="temp.png", 
                                            avg_denom=10
                                           ):

    """
    NOTE: model error (vs target) will ALWAYS be included - predictor, model output, and target are all optional plots! However, they are NOT optional data to include!
    Output plot's shape will depend on the number of fields included, up to 4 plots
    Order of plots (if included) = predictor, model output, truth, error (as before)

    This is not a very well constructed function, and there has to be a more refined and better way to do the dynamic plots, but this works for now
    
    Inputs:
        - predictor --> predictor input (i.e. 2.5 km HRRR for our purposes) from get_model_output
            > If using this function to plot Smartinit, make sure the call to get_model_output_at_idx has crop_pred=False 
        - model_output --> Model output data from get_model_output. CAN ALSO BE SMARTINIT DATA in which case, make sure use_smartinit_mask=True
            > If using this function to plot Smartinit, make sure the call to get_model_output_at_idx has crop_model_output=False 
        - target --> truth (i.e. URMA for our purposes) from get_model_output
            > If using this function to plot Smartinit, make sure the call to get_model_output_at_idx has crop_targ=False 
        - include_predictor --> bool to control if a plot showing the predictor data is included
        - include_model_output --> " " for model output data
        - include_target --> " " for target data
        - use_hrrr_mask --> bool to control if a mask to exclude the NaN regions of the regridded HRRR data will be used. 
            > For Smartinit, it should pretty much always be True, if the goal is to compare Smartinit against HRRRR
        - use_smartinit_mask --> " " for regridded Smartinit data. Doesn't always need to be True, but should be if comparing HRRR against Smartinit
        - date_str --> string of format dt.datetime.strptime(str(np.datetime_as_string(date, unit='s')), "%Y-%m-%dT%H:%M:%S") from get_model_output()
        - title --> model name/params/whatever to identify that plot
        - save_fig --> bool for saving; if True, saves to directory this script is called from (currently this function is not intended for formalized plot saving)
        - save_dir --> master save directory
        - fig_savename --> string for file savename, if to_save = True. Should include ".png" at the end
        - error_units --> string (NOT including parentheses) for variable/error units, e.g. "deg K"
        - avg_denom --> int for how much to scale error plot by. Should be ~10 normally, but for pressurf, should be ~150
    """

    if use_hrrr_mask and not use_smartinit_mask:
        # Plotting only HRRR/URMA vs ML model output. This is to crop the output and URMA, assuming a fill value is used for the boundary
        target = crop_input(target, predictor) 
        model_output = crop_input(model_output, predictor)
        predictor = crop_input(predictor, predictor)
    elif use_smartinit_mask and not use_hrrr_mask: 
        #Plotting only HRRR/URMA vs Smartinit. In this case, model_output is assumed to be Smartinit - if plotting ML model output only in the Smartinit region is desired, then enable both use_hrrr_mask and use_smartinit_mask
        target = crop_input(target, model_output) 
        predictor = crop_input(predictor, model_output)
        model_output = crop_input(model_output, model_output)
    elif use_smartinit_mask and use_hrrr_mask: 
        #Plotting only the overlap region. Here, predictor serves as the HRRR mask, but model_output may not be Smartinit data, so a new instance of Smartinit is called to be safe
        xr_smartinit = get_smartinit_output_at_idx(i=0, target_var='t2m') #only need the mask, don't care about the data
        # First pass thru crop_input to get all data and smartinit mask to the proper HRRR area. Use predictor as the mask, since model_output may have fill values
        smartinit_mask = crop_input(xr_smartinit.data, predictor)
        target = crop_input(target, predictor) 
        model_output = crop_input(model_output, predictor)
        predictor = crop_input(predictor, predictor)
        # Second pass thru crop_input to get all data to the union of the HRRR and smartinit grids
        target = crop_input(target, smartinit_mask)
        predictor = crop_input(predictor, smartinit_mask)
        model_output = crop_input(model_output, smartinit_mask)


    number_of_plots = 1+int(include_predictor)+int(include_model_output)+int(include_target) 
    maxtemp = np.nanmax([np.nanmax(predictor.squeeze()), np.nanmax(model_output.squeeze()), np.nanmax(target.squeeze())])
    mintemp = np.nanmin([np.nanmin(predictor.squeeze()), np.nanmin(model_output.squeeze()), np.nanmin(target.squeeze())])

    avg = (maxtemp-mintemp)/avg_denom #Denominator chosen arbitrarily; adjust if needed
    
    fig, axes = plt.subplots(number_of_plots, 1, figsize=(10, 5.5*number_of_plots))

    

    #Break it down into cases. This flag structure is horrible and should be redone
    pred_flag = True
    model_flag = True
    
    if number_of_plots > 1:
        for i, ax in enumerate(axes[:-1]):
            if include_predictor and pred_flag:
                pos = ax.imshow(predictor.squeeze(), cmap="coolwarm", vmin = mintemp, vmax = maxtemp, origin='lower')
                ax.set_title(f"Predictor (HRRR 2.5km)")
                cbar = fig.colorbar(pos, ax=ax, fraction=0.0225, pad=0.01)
                cbar.set_label(f'{error_units}')
                pred_flag = False #skip this case in the next iteration, if there is one
            elif include_model_output and model_flag:
                pos = ax.imshow(model_output.squeeze(), cmap="coolwarm", vmin = mintemp, vmax = maxtemp, origin='lower')
                ax.set_title(f"Predicted")
                cbar = fig.colorbar(pos, ax=ax, fraction=0.0225, pad=0.01)
                cbar.set_label(f'{error_units}')
                model_flag = False
            elif include_target: #No flag needed here, as this will always be the last plot, i.e. end of for loop
                pos = ax.imshow(target.squeeze(), cmap="coolwarm", vmin = mintemp, vmax = maxtemp, origin='lower')
                ax.set_title(f"Truth (URMA)")
                cbar = fig.colorbar(pos, ax=ax, fraction=0.0225, pad=0.01)
                cbar.set_label(f'{error_units}')

            ax.axis("off")
                
        #Always make the last plot the error
        pos = axes[-1].imshow((model_output.squeeze() - target.squeeze()), cmap='coolwarm', origin='lower', vmin=-1*avg, vmax=avg)
        axes[-1].set_title(f"Prediction - Truth (RMSE = {np.sqrt(np.nanmean((model_output.squeeze() - target.squeeze())**2)):.3f})")
        axes[-1].axis("off")
        cbar = fig.colorbar(pos, ax=axes[-1], fraction=0.0225, pad=0.01)
        cbar.set_label(f'Error ({error_units})')

    #If only the error plot is called, axes is not subscriptable
    if number_of_plots == 1:
        pos = axes.imshow((model_output.squeeze() - target.squeeze()), cmap='coolwarm', origin='lower', vmin=-1*avg, vmax=avg)
        axes.set_title(f"Prediction - Truth (RMSE = {np.sqrt(np.nanmean((model_output.squeeze() - target.squeeze())**2)):.3f})")
        axes.axis("off")
        cbar = fig.colorbar(pos, ax=axes, fraction=0.0225, pad=0.01)
        cbar.set_label(f'Error ({error_units})')
    
    plt.suptitle(f"{title} \n Date = {date_str} \n Maximum = {maxtemp:.1f} | Minimum = {mintemp:.1f}", va="bottom", fontsize=14)
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(f"{save_dir}/{fig_savename}",dpi=300, bbox_inches="tight")
    
    plt.show()
    
    return
