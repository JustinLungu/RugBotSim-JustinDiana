import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from webots_log_processor import WebotsProcessor
from utils import *




folder = "/home/thiemenrug/Documents/_temp/parallel_1/Instance_1/"
x = WebotsProcessor(folder=folder,filename = "webots_log_1.txt",threshold=0.5)


t,y = x.compute_average_belief_over_time()

plt.figure()
plt.plot(t,y)
plt.show()

