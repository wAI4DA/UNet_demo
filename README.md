# How to run UNet_demo
### 1. Clone the UNet_demo repo
You may use `/scratch5/purged` disk space for this test
```
git clone https://github.com/wAI4DA/UNet_demo.git
```
### 2. Train the model
```
cd UNet-demo
vi model_training_sbatch
    # most folks cannot access the gpu QoS, so change gpu-wizard to a normal account,
    # such as wrfruc, zrtrr, etc, and change "#SBATCH -q gpu" to "#SBATCH -q gpuwf"
sbatch model_training_sbatch
```
Once the job is running, you will see two output files as follows:
- `training_log_BS360_NE25.txt`: epochs, loss, and time per epoch
- `JOB_LOG_UNet_training_5473835.out`: job output, including the batch progresses in each epoch    

Ignore the following `PermissionError`:    
```
PermissionError: [Errno 13] Permission denied: '/scratch3/BMC/wrfruc/aschein/Train_Test_Files/train_urma_alltimes_CONUS_t2m.grib2.5b7b6.idx'
```
It just means that ecCodes cannot update the exsiting idx file which is not needed and this does not affect the training process.

### 3. View the trained model
This step will need to set up the Jupyter Lab over SSH on Ursa.    
[This wiki](https://github.com/pyDAmonitor/pyDAmonitor/wiki/Use-Jupyter-Lab-over-SSH-on-Ursa,-Hera,-Jet,-Gaea) can be referred as to how to correctly set up Jupyter Lab.     
The following summarizes the steps for this demo:
#### 3.1. login to Ursa using port forwarding
`ssh -X -L40894:localhost:40894 First.Last@gaea-rsa.boulder.rdhpcs.noaa.gov`  # replace `40894` with your own local port number
#### 3.2.  Request interactive session on GPU nodes
```
account=wrfruc   # gpu-wizard
QoS=gpuwf        # gpu
salloc -A ${account}  -t 3:00:00 -p u1-h100 --mem=0 -q ${QoS}  -N 1 -n 24 --gres=gpu:h100:2
```
When allocated, write down the host name
or run `hostname` and write down the output, such as `u20g01`
#### 3.3. start JupyterLab on the GPU node
```
wget https://raw.githubusercontent.com/wAI4DA/python_env_ai4da/refs/heads/main/load_ai4da.sh
source load_ai4da.sh
jupyter lab --no-browser --port=8820   # change the port number as each user needs to use a different port number
                                       # check the final port number the system allocates as the request one may not be available
```
#### 4.4.  Connect the Ursa front node port to the GPU node port
Connect to an Ursa front node in a new terminal and then run the following command:    
`ssh -L 8820:localhost:8820 u20g01`    # replace `8820` and `u20g01` with your situation    

#### 4.5. Connect to the Jupyter Lab server from your local terminal
`ssh -N -f -p 40894 -L 8820:localhost:8820 First.Last@localhost`    # replace `40894` and `8820` with your situation    

#### 4.6. Open a browser and enter the URL address outputted in step 3.3
eg: `http://localhost:8806/?token=95fa2d9543d9acd01e8c3c9c82ff6b2cd8df3cd341c53ad6`    

#### 4.7.  Open UNet_viewing.ipynb
