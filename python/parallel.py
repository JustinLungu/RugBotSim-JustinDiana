import numpy as np
from webots import WebotsEvaluation
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
import concurrent.futures
from utils import *
import os
import shutil

def remove_dir(directory):
    # Method to delete the run directory    
    if os.path.exists(directory):
        shutil.rmtree(directory)
        
    else:
        raise FileNotFoundError(f"Source directory not found at {directory}")





def launch_instance(x,reevaluation = 0,run_dir=0,feedback =0,fill_ratio = 0.48,n_robots= 5,grid = None,gridsize = 5,desc = "None"):
    instance = reevaluation
    if fill_ratio > .5:
        right_dec=1
    else:
        right_dec = 0
    job = WebotsEvaluation(run = run_dir,instance = reevaluation,robots=n_robots)
    
    world_creation_seed = instance 
    grid_seed = instance + grid_start_seed
    c_settings = {"gamma0":x[0],"gamma":x[1],"tau":x[2],"thetaC":x[3],"swarmCount":x[4],"feedback":feedback,'eta':eta,"seed":instance+grid_start_seed,"sample_strategy":sample_strategy,"size":gridsize,"Usp":Usp,"P(FP)":P_fp,"P(FN)":P_fn}
    s_settings = {"right_dec":right_dec,"fill_ratio":fill_ratio,"offset_f":0.04,"check_interval":10,"autoexit":1,"run_full":run_full}
    settings = {"reevaluation":reevaluation,"word_creation_seed":world_creation_seed,"grid_seed":grid_seed ,"description":desc}

    job.job_setup(c_settings=c_settings,s_settings=s_settings,settings=settings,world_creation_seed=world_creation_seed,grid_seed=grid_seed,fill_ratio=fill_ratio,gridsize=gridsize,grid_ = grid)
    job.run_webots_instance(port=1234+instance)
    fitness = job.get_fitness()
    #replace lower directory
    job.move_results("/home/thiemenrug/Documents/_temp/",f"parallel_{run_dir}/Instance_{instance}")
    job.remove_run_dir()
    del x
    return fitness


def launch_batch(batch_size,workers,x_,run_dir,feedback,fill_ratio,robots,grid=None,size = 5,desc="None"):
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as process_executor:  
        futures = [process_executor.submit(launch_instance, x_,i,run_dir,feedback,fill_ratio,robots,grid,size,desc) for i in range(batch_size)]
        [future.result() for future in futures]
    remove_dir(f"jobfiles/Run_{run_dir}")


M1 = diagonal_matrix()
M2 = stripe_matrix()
M3 = block_diagonal_matrix()
M4 = organized_alternating_matrix()
M5 = random_matrix()


grid_start_seed = 1 # this is for the seed of the pattern
sample_strategy = 0 # your method
P_fp = 0
P_fn = 0

# soft feedback
eta=2
Usp = 0
## end soft feedback

#select feedback strategy
#0 for Umin, 1 Uplus, 2 for soft feedback


#run parameters
#run_full =1 for running the full time, =0 for quiting on decision made
run_full =0

x = [7500,15000,2000,55,380] #[\gamma0,\gamma,\tau,\Theta_c CA,swarmCount]

launch_batch(100,7,x,1,0,0.48,5)