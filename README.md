# How to run UNet_demo
### 1. Clone the UNet_demo repo
You may use `/scratch5/purged` disk space for this test
```
git clone https://github.com/wAI4DA/UNet_demo.git
```
### 2. Train the model
```
cd UNet-demo
vi model_training_sbatch.sh
    # most folks cannot access the gpu QoS, so change gpu-wizard to a normal account,
    # such as wrfruc, zrtrr, etc, and change "#SBATCH -q gpu" to "#SBATCH -q gpuwf"
sbatch model_training_sbatch.sh
```
Once the job runs, you will see two output files as follows:
- `training_log_BS360_NE25.txt`: epochs, loss, and time per epoch
- `JOB_LOG_UNet_training_5473835.out`: job output, including the batch progresses in each epoch    

Ignore the following `PermissionError`:    
```
PermissionError: [Errno 13] Permission denied: '/scratch3/BMC/wrfruc/aschein/Train_Test_Files/train_urma_alltimes_CONUS_t2m.grib2.5b7b6.idx'
```
It just means that ecCodes cannot update the exsiting idx file, which is not needed and these "errors" do not affect the training process.

### 3. View the trained model
We will need to set up the Jupyter Lab over SSH on Ursa in order to view the trained model.    
You are recommeneded to check [this wiki](https://github.com/pyDAmonitor/pyDAmonitor/wiki/Use-Jupyter-Lab-over-SSH-on-Ursa,-Hera,-Jet,-Gaea) on how to correctly set up Jupyter Lab.     
The following summarizes the example steps for this UNet_demo:
#### 3.1. login to Ursa using port forwarding
`ssh -X -L40894:localhost:40894 First.Last@ursa-rsa.boulder.rdhpcs.noaa.gov`  # replace `40894` with your own local port number
#### 3.2.  Request an interactive session on GPU nodes
```
account=wrfruc   # gpu-wizard
QoS=gpuwf        # gpu
salloc -A ${account}  -t 4:00:00 -p u1-h100 --mem=0 -q ${QoS}  -N 1 -n 24 --gres=gpu:h100:2 --mail-type=BEGIN --mail-user=Guoqing.Ge@noaa.gov
      # Replace with your own email so that you will get an email notification when a GPU node is allocated, which will take quite a while
```
When allocated, write down the host name
or run `hostname` and write down the output, such as `u20g01`, which will be used in step 3.4
#### 3.3. start JupyterLab on the GPU node
```
source /scratch3/BMC/wrfruc/gge/AI/ai4da/load_ai4da.sh
jupyter lab --no-browser --port=8820   # change the port number as each user needs to use a different port number
                                       # check the final port number the system allocates as it may not be your requested number
```
#### 3.4.  Connect the Ursa front node port to the GPU node port
Start a new terminal windown, connect to Ursa, and then run the following command:    
`ssh -L 8820:localhost:8820 u20g01`    # replace `8820` and `u20g01` with your situation    

#### 3.5. Connect to the Jupyter Lab server from your local terminal
`ssh -N -f -p 40894 -L 8820:localhost:8820 First.Last@localhost`    # replace `40894` and `8820` with your situation    
No output if succeeded, but will get errors if failed.

#### 3.6. Open a browser and enter the URL address outputted in step 3.3
eg: `http://localhost:8820/?token=95fa2d9543d9acd01e8c3c9c82ff6b2cd8df3cd341c53ad6`    

#### 3.7.  Open UNet_viewing.ipynb
You may want to change `filename` to your trained model.     
Note: need to remove the file name suffix `.pt`.    
eg: `filename = f"residual_attn_CONUS_BS360_NE25_tD_pred(t2m)_targ(t2m)"`
