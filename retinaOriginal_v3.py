import nest, numpy, os, sys
import matplotlib.pyplot as plt
from makeGifs import gifMaker
nest.sli_run('M_WARNING setverbosity') # avoid writing too many NEST messages
nest.ResetKernel()
nest.SetKernelStatus({'resolution': 0.01, 'local_num_threads':4, 'print_time': True})

# To do:
# all interneurons non spiking
# voltage simulation instead of current


##########################
### Set the parameters ###
##########################

# Simulation parameters
simulationTime = 100.0      # [ms]
stepDuration   =   1.0      # [ms]  # put 1.0 here to see nice gifs
startTime      =   0.0      # [ms]
stopTime       =  10.0      # [ms]

# Retina parameters
BGCRatio       =    4
inhibRangeHC   =    2       # [pixels]
inhibRangeAC   =    2       # [pixels]
nRows          =   20       # [pixels]
nCols          =   20       # [pixels]

# Input parameters
inputTarget    =   (10, 10)  # [pixels]
inputRadius    =    5       # [pixels]
inputGain      = 2000.0     # [pA]
inputNoise     =   10.0     # [pA]

# Set the neurons whose LFP is going to be recorded
neuronsToRecord = [(inputTarget[0]+  0,           inputTarget[1]+0),
                   (inputTarget[0]+  1,           inputTarget[1]+0),
                   (inputTarget[0]+  inputRadius, inputTarget[1]+0)]
                   # (inputTarget[0]+2*inputRadius, inputTarget[1]+0)]

# Neurons custom parameters
threshPot         = -55.0
restPot           = -70.0  # should be 'E_l' but not sure
neuronModel       = 'iaf_cond_alpha'
neuronParams      = {'V_th': threshPot,      'tau_syn_ex': 1.0, 'V_reset': -70.0}
interNeuronParams = {'V_th': threshPot+1000, 'tau_syn_ex': 1.0, 'V_reset': -70.0}

# Connection parameters
connections    = {
    'BC_To_GC' :  500.0*1,  # [nS/spike]
    'AC_To_GC' : -100.0*2,  # [nS/spike]
    'HC_To_BC' : -100.0*2,  # [nS/spike]
    'BC_To_AC' :   50.0*1}  # [nS/spike]

# Scale the weights, if needed
weightScale    = 0.0005
for key, value in connections.items():
    connections[key] = value*weightScale


#########################
### Build the neurons ###
#########################

# Excitatory cells
GC = nest.Create(neuronModel,          nRows*nCols,      neuronParams)
BC = nest.Create(neuronModel, BGCRatio*nRows*nCols, interNeuronParams)

# Inhibitory cells
nHCRows = int(float(nRows)/float(inhibRangeHC))+1
nHCCols = int(float(nCols)/float(inhibRangeHC))+1
nACRows = int(float(nRows)/float(inhibRangeAC))+1
nACCols = int(float(nCols)/float(inhibRangeAC))+1
AC = nest.Create(neuronModel, nACRows*nACCols, interNeuronParams)
HC = nest.Create(neuronModel, nHCRows*nHCCols, interNeuronParams)

# Spike detectors
GCSD = nest.Create('spike_detector',            nRows*  nCols)
# BCSD = nest.Create('spike_detector', BGCRatio*  nRows*  nCols)
# ACSD = nest.Create('spike_detector',          nACRows*nACCols)
# HCSD = nest.Create('spike_detector',          nHCRows*nHCCols)

# Connect the spike detectors to their respective populations
nest.Connect(GC, GCSD, 'one_to_one')
# nest.Connect(BC, BCSD, 'one_to_one')
# nest.Connect(AC, ACSD, 'one_to_one')
# nest.Connect(HC, HCSD, 'one_to_one')

# Create the gif makers, for each population
gifMakerList = []
# gifMakerList.append(gifMaker(name='GC', popID=GCSD, dimTuple=(1,            nRows,   nCols), orientedMap=False, spiking=True , baseline=None   ))
# gifMakerList.append(gifMaker(name='BC', popID=BC,   dimTuple=(1, BGCRatio,  nRows,   nCols), orientedMap=True , spiking=False, baseline=restPot))
# gifMakerList.append(gifMaker(name='AC', popID=AC,   dimTuple=(1,          nACRows, nACCols), orientedMap=False, spiking=False, baseline=restPot))
# gifMakerList.append(gifMaker(name='HC', popID=HC,   dimTuple=(1,          nHCRows, nHCCols), orientedMap=False, spiking=False, baseline=restPot))


# # Create and connect the multimeter to simulate interneurons
# BCMMs = []
# ACMMs = []
# HCMMs = []
#
# # Bipolar cells multimeters (vesicles fusion proportionnal to their potential)
# for i in range(len(BCSD)):
#
# 	BCMM = nest.Create('multimeter', params = {'withtime': True, 'interval': stepDuration, 'record_from': ['V_m']})
# 	nest.Connect(BCMM, [BC[i]])
# 	BCMMs.append(BCMM)
#
# # Amacrine cells multimeters (vesicles fusion proportionnal to their potential)
# for i in range(len(neuronsToRecord)):
#
# 	ACMM = nest.Create('multimeter', params = {'withtime': True, 'interval': stepDuration, 'record_from': ['V_m']})
# 	nest.Connect(ACMM, [AC[i]])
# 	ACMMs.append(ACMM)
#
# # Horizontal cells multimeters (vesicles fusion proportionnal to their potential)
# for i in range(len(neuronsToRecord)):
#
# 	HCMM = nest.Create('multimeter', params = {'withtime': True, 'interval': stepDuration, 'record_from': ['V_m']})
# 	nest.Connect(HCMM, [HC[i]])
# 	HCMMs.append(HCMM)

# Create and connect the multimeter to plot
GCMMs = []
for i in range(len(neuronsToRecord)):

    GCMM = nest.Create('multimeter', params = {'withtime': True, 'interval': 0.01, 'record_from': ['V_m']})
    nest.Connect(GCMM, [GC[0*nRows*nCols + neuronsToRecord[i][0]*nCols + neuronsToRecord[i][1]]])
    GCMMs.append(GCMM)

BCMMs = []
for i in range(len(neuronsToRecord)):

    BCMM = nest.Create('multimeter', params = {'withtime': True, 'interval': 0.01, 'record_from': ['V_m']})
    nest.Connect(BCMM, [BC[0*nRows*nCols + neuronsToRecord[i][0]*nCols + neuronsToRecord[i][1]]])
    BCMMs.append(BCMM)

ACMMs = []
for i in range(len(neuronsToRecord)):

    ACMM = nest.Create('multimeter', params = {'withtime': True, 'interval': 0.01, 'record_from': ['V_m']})
    row  = int(float(neuronsToRecord[i][0])/float(inhibRangeAC))
    col  = int(float(neuronsToRecord[i][1])/float(inhibRangeAC))
    nest.Connect(ACMM, [AC[row*nACCols + col]])
    ACMMs.append(ACMM)

HCMMs = []
for i in range(len(neuronsToRecord)):

    HCMM = nest.Create('multimeter', params = {'withtime': True, 'interval': 0.01, 'record_from': ['V_m']})
    row  = int(float(neuronsToRecord[i][0]/inhibRangeHC))
    col  = int(float(neuronsToRecord[i][1]/inhibRangeHC))
    nest.Connect(HCMM, [HC[row*nHCCols + col]])
    HCMMs.append(HCMM)

# # Create and connect the multimeter to plot
# ACMMs = []
# for i in range(len(neuronsToRecord)):
#
# 	ACMM = nest.Create('multimeter', params = {'withtime': True, 'interval': 0.01, 'record_from': ['V_m', 'g_ex', 'g_in']})
# 	nest.Connect(ACMM, [AC[0*nRows*nCols + neuronsToRecord[i][0]*nCols + neuronsToRecord[i][1]]])
# 	ACMMs.append(ACMM)



##############################################
### Set the input and simulate the network ###
##############################################

# Make the current stimulus directory
figureDir = 'SimFigures'
if not os.path.exists(figureDir):
    os.makedirs(figureDir)

# Simulate the network
timeSteps = int(simulationTime/stepDuration)
for time in range(timeSteps):

    # Start the stimulus
    if time == int(startTime/stepDuration):

        # Ganglion cells
        for i in range(nRows):
            for j in range(nCols):
                distance = numpy.sqrt((i-inputTarget[0])**2 + (j-inputTarget[1])**2)
                if distance < inputRadius:
                    noiseGain     = inputNoise*(numpy.random.rand()-0.5)*2.0
                    inputStrength = (inputGain+noiseGain)*numpy.sqrt((inputRadius-distance)/inputRadius)
                    nest.SetStatus([GC[i*nRows + j]], {'I_e': inputStrength})

        # Bipolar cells input
        for i in range(nRows):
            for j in range(nCols):
                distance = numpy.sqrt((i-inputTarget[0])**2 + (j-inputTarget[1])**2)
                if distance < inputRadius:
                    noiseGain     = inputNoise*(numpy.random.rand()-0.5)*2.0
                    inputStrength = (inputGain+noiseGain)*numpy.sqrt((inputRadius-distance)/inputRadius)
                    for k in range(BGCRatio):
                        nest.SetStatus([BC[k*nRows*nCols + i*nRows + j]], {'I_e': inputStrength*0.5})

        # Amacrine cells input
        for iAC in range(nACRows):
            for jAC in range(nACCols):
                distance = numpy.sqrt((inhibRangeAC*iAC-inputTarget[0])**2 + (inhibRangeAC*jAC-inputTarget[1])**2)
                if distance < inputRadius:
                    noiseGain     = inputNoise*(numpy.random.rand()-0.5)*2.0
                    inputStrength = (inputGain+noiseGain)*numpy.sqrt((inputRadius-distance)/inputRadius)
                    nest.SetStatus([AC[iAC*nACRows + jAC]], {'I_e': inputStrength*0.8})

        # Horizontal cells input
        for iHC in range(nHCRows):
            for jHC in range(nHCCols):
                distance = numpy.sqrt((inhibRangeHC*iHC-inputTarget[0])**2 + (inhibRangeHC*jHC-inputTarget[1])**2)
                if distance < inputRadius:
                    noiseGain     = inputNoise*(numpy.random.rand()-0.5)*2.0
                    inputStrength = (inputGain+noiseGain)*numpy.sqrt((inputRadius-distance)/inputRadius)
                    nest.SetStatus([HC[iHC*nHCRows + jHC]], {'I_e': inputStrength*0.4})

    # Stop the stimulus
    if time == int(stopTime/stepDuration):
        nest.SetStatus(GC, {'I_e': 0.0})
        nest.SetStatus(BC, {'I_e': 0.0})
        nest.SetStatus(AC, {'I_e': 0.0})
        nest.SetStatus(HC, {'I_e': 0.0})

    # Connections from bipolar cells to the retinal ganglion cells
    source = []
    target = []
    for i in range(nRows):
        for j in range(nCols):
            for k in range(BGCRatio):
                source = (k*nRows*nCols + i*nCols + j)
                target = (                i*nCols + j)
                preSynVoltage  = nest.GetStatus([BC[source]], 'V_m')[0] - restPot
                if i == inputTarget[0] and j == inputTarget[1]:
                    postSynVoltage = nest.GetStatus([GC[target]], 'V_m')[0] - restPot
                    nest.SetStatus([GC[target]], {'V_m': restPot + postSynVoltage + connections['BC_To_GC']*preSynVoltage})

    # Connections from amacrine cells to ganglion cells
    source = []
    target = []
    for i in range(-inhibRangeAC, inhibRangeAC+1):
        for j in range(-inhibRangeAC, inhibRangeAC+1):
            if i != 0 and j != 0:
                distance = numpy.sqrt(i**2 + j**2)
                inhibACWeight = connections['AC_To_GC']/distance
                for iAC in range(nACRows):
                    for jAC in range(nACCols):
                        convertedRow = inhibRangeAC*iAC + i
                        convertedCol = inhibRangeAC*jAC + j
                        if 0 <= convertedRow < nRows and 0 <= convertedCol < nCols:
                            source = (iAC       *nACCols + jAC         )
                            target = (convertedRow*nCols + convertedCol)
                            preSynVoltage  = nest.GetStatus([AC[source]], 'V_m')[0] - restPot
                            postSynVoltage = nest.GetStatus([GC[target]], 'V_m')[0] - restPot
                            nest.SetStatus([GC[target]], {'V_m': restPot + postSynVoltage + connections['AC_To_GC']*preSynVoltage})

    # Connections from horizontal cells to bipolar cells
    source = []
    target = []
    for i in range(-inhibRangeHC, inhibRangeHC+1):
        for j in range(-inhibRangeHC, inhibRangeHC+1):
            if i != 0 and j != 0:
                distance = numpy.sqrt(i**2 + j**2)
                inhibHCWeight = connections['HC_To_BC']/distance
                for iHC in range(nHCRows):
                    for jHC in range(nHCCols):
                        convertedRow = inhibRangeHC*iHC + i
                        convertedCol = inhibRangeHC*jHC + j
                        if 0 <= convertedRow < nRows and 0 <= convertedCol < nCols:
                            for k in range(BGCRatio):
                                source = (                iHC       *nHCCols + jHC         )
                                target = (k*nRows*nCols + convertedRow*nCols + convertedCol)
                                preSynVoltage  = nest.GetStatus([HC[source]], 'V_m')[0] - restPot
                                postSynVoltage = nest.GetStatus([BC[target]], 'V_m')[0] - restPot
                                nest.SetStatus([BC[target]], {'V_m': restPot + postSynVoltage + connections['HC_To_BC']*preSynVoltage})

    # Connections from bipolar cells to amacrine cells
    source = []
    target = []
    inhibACWeight = connections['AC_To_GC']/numpy.sqrt(i**2 + j**2)
    for iAC in range(nACRows):
        for jAC in range(nACCols):
            if 0 <= inhibRangeAC*iAC < nRows and 0 <= inhibRangeAC*jAC < nCols:
                for k in range(BGCRatio):
                    source = (k*nRows*nCols + inhibRangeAC*iAC*nCols   + inhibRangeAC*jAC)
                    target = (                             iAC*nACCols +              jAC)
                    preSynVoltage  = nest.GetStatus([BC[source]], 'V_m')[0] - restPot
                    postSynVoltage = nest.GetStatus([AC[target]], 'V_m')[0] - restPot
                    nest.SetStatus([AC[target]], {'V_m': restPot + postSynVoltage + connections['BC_To_AC']*preSynVoltage})

    # Run the simulation for one gif frame
    nest.Simulate(stepDuration)
    if time < timeSteps-1:
        sys.stdout.write("\033[2F") # move the cursor back to previous line

    # Take screenshots of every recorded population
    for instance in gifMakerList: # gifMaker.getInstances():
        (namePop, nSpikes) = instance.takeScreenshot()


#################################
### Read the network's output ###
#################################

# Create animated gif of stimulus
sys.stdout.write('Creating animated gifs.\n\n')
sys.stdout.flush()
for instance in gifMakerList: # gifMaker.getInstances():
    instance.createGif(figureDir, durationTime=0.2)

for i in range(len(neuronsToRecord)):

    # Obtain and display data
    recRow = neuronsToRecord[i][0]
    recCol = neuronsToRecord[i][1]
    spikes = nest.GetStatus([GCSD[recRow*nRows+recCol]], keys='events')[0]['times']

    # Do the rasterplot
    plt.subplot(5, len(neuronsToRecord), i+1)
    plt.plot([startTime, stopTime], [1.25, 1.25], 'c-', lw=4)
    for spike in spikes:
        plt.plot([spike, spike], [0, 1], 'k-', lw=2)
    plt.axis([0, simulationTime, 0, 1.5])
    plt.ylabel('Rasterplot')

    # Plot the membrane potential of GC
    events = nest.GetStatus(GCMMs[i])[0]['events']
    tPlot  = events['times'];
    plt.subplot(5, len(neuronsToRecord), len(neuronsToRecord)+i+1)
    plt.plot(tPlot, events['V_m'])
    plt.plot([0, simulationTime], [threshPot, threshPot], 'k-', lw=1)
    plt.axis([0, simulationTime, -100, -25])
    plt.ylabel('GC membr. pot. [mV]')

    # Plot the membrane potential of BC
    events = nest.GetStatus(BCMMs[i])[0]['events']
    tPlot  = events['times'];
    plt.subplot(5, len(neuronsToRecord), 2*len(neuronsToRecord)+i+1)
    plt.plot(tPlot, events['V_m'])
    plt.plot([0, simulationTime], [restPot, restPot], 'k-', lw=1)
    plt.axis([0, simulationTime, -100, -25])
    plt.ylabel('BC membr. pot. [mV]')

    # Plot the membrane potential of AC
    events = nest.GetStatus(ACMMs[i])[0]['events']
    tPlot  = events['times'];
    plt.subplot(5, len(neuronsToRecord), 3*len(neuronsToRecord)+i+1)
    plt.plot(tPlot, events['V_m'])
    plt.plot([0, simulationTime], [restPot, restPot], 'k-', lw=1)
    plt.axis([0, simulationTime, -100, -25])
    plt.ylabel('AC membr. pot. [mV]')

    # Plot the membrane potential of HC
    events = nest.GetStatus(HCMMs[i])[0]['events']
    tPlot  = events['times'];
    plt.subplot(5, len(neuronsToRecord), 4*len(neuronsToRecord)+i+1)
    plt.plot(tPlot, events['V_m'])
    plt.plot([0, simulationTime], [restPot, restPot], 'k-', lw=1)
    plt.axis([0, simulationTime, -100, -25])
    plt.ylabel('HC membr. pot. [mV]')

# Show the plot
plt.show()
