import numpy
import matplotlib.pyplot as plt
from utilities import *

time  =  numpy.arange(0, 70, 1) #

timeSteps = int(70/1)
for t in range(timeSteps):
    Pattern = voltagePattern(500,t,0,10,'triangle')

plt.plot(time, Pattern)
plt.show()
