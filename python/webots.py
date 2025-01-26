import os
import subprocess
import time
from webotsWorldCreation import createWorld

class WebotsEvaluation():

    def __init__(self):
        self.run =1
        self.instance =1
        self.robots = 4
        pass
    def run_webots_instance(self):
        # Method to run a Webots instance with given parameters
        subprocess.check_call(['.././run_webots.sh', str(self.run), str(self.instance), 
                               str(self.robots)])
        print(os.getcwd())
        os.chdir("../../")
        print(os.getcwd())

    def job_setup(self,c_settings= {},s_settings = {},settings={}):
        

        run_dir = f"jobfiles/Run_{self.run}/"
        os.makedirs(os.path.dirname(f"{run_dir}Instance_{self.instance}/"), exist_ok=True)

        world = createWorld(self.instance,self.instance,f"world_{self.instance}",self.robots)
        world.save_settings(run_dir,c_settings,s_settings)
        world.saveGrid(run_dir=run_dir)
        world.create_world()
        setup =  f"{run_dir}Instance_{self.instance}/settings.txt"

        with open(setup, 'w') as file:
            for value in [settings.values()]:
                file.write(str(value) + '\n')
        time.sleep(1)

        os.chdir(run_dir)  # Changing directory to the run directory
        
        pass

robots= [4]
learningrates =[1]
for i in range(1):

    x = WebotsEvaluation()
    x.instance = 1
    x.robots = 1
    c_settings = {}
    s_settings = {}
    x.job_setup(c_settings=c_settings,s_settings=s_settings)
    x.run_webots_instance()
    
    del x