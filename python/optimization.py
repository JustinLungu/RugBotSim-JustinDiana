import numpy as np
from webots import WebotsEvaluation
from PSO import PSO


def rosenbrock(x,particle=0,iteration=0,reevaluation=0): 
    value = sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    #value = value - (len(x)-1)
    return value 
def cost_function(x,particle=0,iteration=0,reevaluation=0):
    run_dir = 2
    job = WebotsEvaluation(run = run_dir,instance = particle * reevaluation,robots=4)
    
    world_creation_seed = (particle+1) * (iteration + 1) * reevaluation
    grid_seed = reevaluation * (iteration +1)
    c_settings = {"gamma0":x[0],"gamma":x[1],"tau":x[2],"thetaC":x[3],"swarmCount":x[4],"feedback":0}
    s_settings = {"right_dec":0,"fill_ratio":0.48,"offset_f":0.04,"check_interval":20,"autoexit":1}
    settings = {"particle":particle,"iteration":iteration,"reevaluation":reevaluation,"word_creation_seed":world_creation_seed,"grid_seed":grid_seed}
    job.job_setup(c_settings=c_settings,s_settings=s_settings,settings=settings,world_creation_seed=world_creation_seed,grid_seed=grid_seed,fill_ratio=0.48,gridsize=5)
    job.run_webots_instance()
    fitness = job.get_fitness()
    job.move_results("/home/thiemenrug/Desktop/",f"pso_{run_dir}/I{iteration}/P{particle}/R{reevaluation}")
    job.remove_run_dir()
    del x

    return fitness


bounds = np.array([[0,20000],[0,20000],[1000,6000],[50,150],[0,500]])
#bounds = np.array([[-100,100],[-100,100]])

pso = PSO(1,0,[1,0.4],0.5,0.5,bounds,0,cost_function,10,10,5,.2,2)
pso.pso_threaded(5)
pso.webots_data.to_csv("jobfiles/pso.csv")