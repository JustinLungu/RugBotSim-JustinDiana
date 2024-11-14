import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from webots_log_processor import WebotsProcessor
from utils import *




folder = "/home/diana/Documents/_temp/parallel_1/Instance_0/"
x = WebotsProcessor(folder=folder,filename = "webots_log_0.txt",threshold=0.5)


t,y = x.compute_average_belief_over_time()

plt.figure()
plt.plot(t,y)
plt.show()

a,b = x.compute_std_beliefs_over_time()

plt.figure()
plt.plot(a,b)
plt.show()