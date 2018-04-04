import nest, numpy, os, sys
import matplotlib.pyplot as plt
from makeGifs import gifMaker
nCoresToUse = 1
nest.sli_run('M_WARNING setverbosity') # avoid writing too many NEST messages
nest.ResetKernel()
nest.SetKernelStatus({'resolution': 0.01, 'local_num_threads':nCoresToUse, 'print_time': True})

# To do:
# all interneurons non spiking
# voltage simulation instead of current


##########################
### Set the parameters ###
##########################

# Simulation parameters
simulationTime =   110.0     # [ms]
stepDuration   =   1.0      # [ms]  # put 1.0 here to see nice gifs
startTime      =   0.0      # [ms]
stopTime       =  10.0      # [ms]

# Retina parameters
BGCRatio        =    4
AGCRatio        =    5
HGCRatio        =    1
excitRangeBC    =    1
inhibRangeHC    =    1       # [pixels]
inhibRangeAC    =    2       # [pixels]
nonInhibRangeHC =    0
nonInhibRangeAC =    1       # [pixels]
RC_GC           =    1       # [ms]
RC_BC           =    10       # [ms]
RC_AC           =    3       # [ms]
RC_HC           =    14       # [ms]
nRows           =   10       # [pixels]
nCols           =   10       # [pixels]

# Input parameters
inputTarget    =   (5, 5)    # [pixels]
inputRadius    =    3        # [pixels]
#inputGain      = 1200.0     # [pA] or [mV]
inputVoltage   =   0.05*180      # [mV]
inputNoise     =   10.0
def inputSpaceFrame(d, sigma):
    return numpy.exp(-d**2/(2*sigma**2))
def inputTimeFrame(RC, gain, t, start, stop):
    if start < t < stop:
        return gain*(1-numpy.exp(-(t-start)/RC))
    if t >= stop:
        return gain*(1-numpy.exp(-(stop-start)/RC))*numpy.exp (-(t-stop)/RC)
    else:
        return 0

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
    'BC_To_GC' :  500.0,  # 100 [nS/spike]
    'AC_To_GC' : -100.0,  # -100 [nS/spike]
    'HC_To_BC' : -10.0 ,  # -25 [nS/spike]
    'BC_To_AC' :  10.0 }  # 10 [nS/spike]

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
#nHCRows = int(float(nRows)/float(inhibRangeHC))+1
#nHCCols = int(float(nCols)/float(inhibRangeHC))+1
#nACRows = int(float(nRows)/float(inhibRangeAC))+1
#nACCols = int(float(nCols)/float(inhibRangeAC))+1
AC = nest.Create(neuronModel, AGCRatio*nRows*nCols, interNeuronParams)
HC = nest.Create(neuronModel, HGCRatio*nRows*nCols, interNeuronParams)

# Spike detectors
GCSD = nest.Create('spike_detector',            nRows* nCols)

# Connect the spike detectors to their respective populations
nest.Connect(GC, GCSD, 'one_to_one')

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
    nest.Connect(ACMM, [AC[0*nRows*nCols + neuronsToRecord[i][0]*nCols + neuronsToRecord[i][1]]])
    ACMMs.append(ACMM)

HCMMs = []
for i in range(len(neuronsToRecord)):

    HCMM = nest.Create('multimeter', params = {'withtime': True, 'interval': 0.01, 'record_from': ['V_m']})
    nest.Connect(HCMM, [HC[0*nRows*nCols + neuronsToRecord[i][0]*nCols + neuronsToRecord[i][1]]])
    HCMMs.append(HCMM)


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
    if int(startTime/stepDuration) < time < int(stopTime/stepDuration):

        # Ganglion cells
        for i in range(nRows):
            for j in range(nCols):
                distance = numpy.sqrt((i-inputTarget[0])**2 + (j-inputTarget[1])**2)
                if distance < inputRadius:
                    noiseGain     = inputNoise*(numpy.random.rand()-0.5)*2.0
                    inputStrength = (inputVoltage+noiseGain)*inputSpaceFrame(distance, 0.5*inputRadius)
                    StimGC= inputTimeFrame(RC_GC, inputStrength, time, startTime, stopTime)
                    target        = (                 i*nCols + j)
                    GCVoltage = nest.GetStatus([GC[target]], 'V_m')[0] - restPot
                    nest.SetStatus([GC[target]], {'V_m': restPot + GCVoltage + StimGC})

        # Bipolar cells input
        for i in range(nRows):
            for j in range(nCols):
                distance = numpy.sqrt((i-inputTarget[0])**2 + (j-inputTarget[1])**2)
                if distance < inputRadius:
                    noiseGain     = inputNoise*(numpy.random.rand()-0.5)*2.0
                    inputStrength = (inputVoltage+noiseGain)*inputSpaceFrame(distance, 0.5*inputRadius)
                    StimBC= inputTimeFrame(RC_BC, inputStrength, time, startTime, stopTime)
                    for k in range(BGCRatio):
                        target    = (k*nRows*nCols + i*nRows + j)
                        BCVoltage = nest.GetStatus([BC[target]], 'V_m')[0]- restPot
                        nest.SetStatus([BC[target]], {'V_m': restPot + BCVoltage + StimBC*0.69})

        # Amacrine cells input
        for i in range(nRows):
            for j in range(nCols):
                distance = numpy.sqrt((i-inputTarget[0])**2 + (j-inputTarget[1])**2)
                if distance < inputRadius:
                    noiseGain     = inputNoise*(numpy.random.rand()-0.5)*2.0
                    inputStrength = (inputVoltage+noiseGain)*inputSpaceFrame(distance, 0.5*inputRadius)
                    StimAC= inputTimeFrame(RC_AC, inputStrength, time, startTime, stopTime)
                    for k in range(AGCRatio):
                        target    = (k*nRows*nCols + i*nRows + j)
                        ACVoltage = nest.GetStatus([AC[target]], 'V_m')[0] - restPot
                        nest.SetStatus([AC[target]], {'V_m': restPot + ACVoltage + StimAC*0.61})

        # Horizontal cells input
        for i in range(nRows):
            for j in range(nCols):
                distance = numpy.sqrt((i-inputTarget[0])**2 + (j-inputTarget[1])**2)
                if distance < inputRadius:
                    noiseGain     = inputNoise*(numpy.random.rand()-0.5)*2.0
                    inputStrength = (inputVoltage+noiseGain)*inputSpaceFrame(distance, 0.5*inputRadius)
                    StimHC= inputTimeFrame(RC_HC, inputStrength, time, startTime, stopTime)
                    for k in range(HGCRatio):
                        target    = (k*nRows*nCols + i*nRows + j)
                        HCVoltage = nest.GetStatus([HC[target]], 'V_m')[0] - restPot
                        nest.SetStatus([HC[target]], {'V_m': restPot + HCVoltage + StimHC*0.53})

    # # Stop the stimulus
    # if time == int(stopTime/stepDuration):
    #     nest.SetStatus(GC, {'I_e': 0.0})
    #     nest.SetStatus(BC, {'I_e': 0.0})
    #     nest.SetStatus(AC, {'I_e': 0.0})
    #     nest.SetStatus(HC, {'I_e': 0.0})

    # Connections from bipolar cells to the retinal ganglion cells
    source = []
    target = []
    for i in range(nRows):
        for j in range(nCols):
            for kBC in range(BGCRatio):
                source = (kBC*nRows*nCols + i*nCols + j)
                target = (                  i*nCols + j)
                preSynVoltage  = nest.GetStatus([BC[source]], 'V_m')[0] - restPot
                # if i == inputTarget[0] and j == inputTarget[1]: ?????
                postSynVoltage = nest.GetStatus([GC[target]], 'V_m')[0] - restPot
                nest.SetStatus([GC[target]], {'V_m': restPot + postSynVoltage + connections['BC_To_GC']*preSynVoltage})

    # Connections from amacrine cells to ganglion cells
    source = []
    target = []
    for i2 in range(-inhibRangeAC, inhibRangeAC+1):
        for j2 in range(-inhibRangeAC, inhibRangeAC+1):
            if numpy.abs(i2) > nonInhibRangeAC and numpy.abs(j2) > nonInhibRangeAC:
                for kAC in range(AGCRatio):
                    for i in range (nRows):
                        for j in range (nCols):
                            if 0 < (i+i2) < nRows and 0 < (j+j2) < nCols:
                                source = (kAC*nRows*nCols + i     *nCols +  j    )
                                target = (                  (i+i2)*nCols + (j+j2))
                                preSynVoltage  = nest.GetStatus([AC[source]], 'V_m')[0] - restPot
                                postSynVoltage = nest.GetStatus([GC[target]], 'V_m')[0] - restPot
                                nest.SetStatus([GC[target]], {'V_m': restPot + postSynVoltage + connections['AC_To_GC']*preSynVoltage})

    # Connections from horizontal cells to bipolar cells
    source = []
    target = []
    for i2 in range(-inhibRangeHC, inhibRangeHC+1):
        for j2 in range(-inhibRangeHC, inhibRangeHC+1):
            if i2 != 0 and j2 != 0:
                for kHC in range(HGCRatio):
                    for kBC in range(BGCRatio):
                        for i in range(nRows):
                            for j in range(nCols):
                                if 0 < (i+i2) < nRows and 0 < (j+j2) < nCols:
                                    source = (kHC*nRows*nCols +  i    *nCols +  j    )
                                    target = (kBC*nRows*nCols + (i+i2)*nCols + (j+j2))
                                    preSynVoltage  = nest.GetStatus([HC[source]], 'V_m')[0] - restPot
                                    postSynVoltage = nest.GetStatus([BC[target]], 'V_m')[0] - restPot
                                    nest.SetStatus([BC[target]], {'V_m': restPot + postSynVoltage + connections['HC_To_BC']*preSynVoltage})

    # Connections from bipolar cells to amacrine cells
    source = []
    target = []
    for i2 in range(-excitRangeBC, excitRangeBC+1):
        for j2 in range(-excitRangeBC, excitRangeBC+1):
            for kAC in range(AGCRatio):
                    for kBC in range(BGCRatio):
                        for i in range(nRows):
                            for j in range(nCols):
                                if 0 < (i+i2) < nRows and 0 < (j+j2) < nCols:
                                    source = (kBC*nRows*nCols +  i    *nCols +  j     )
                                    target = (kAC*nRows*nCols + (i+i2)*nCols + (j+j2) )
                                    preSynVoltage  = nest.GetStatus([BC[source]], 'V_m')[0] - restPot
                                    postSynVoltage = nest.GetStatus([AC[target]], 'V_m')[0] - restPot
                                    nest.SetStatus([AC[target]], {'V_m': restPot + postSynVoltage + connections['BC_To_AC']*preSynVoltage})

    # Run the simulation for one gif frame
    nest.Simulate(stepDuration)
    # if time < timeSteps-1:
    #     sys.stdout.write("\033[2F") # move the cursor back to previous line

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

    # Plot the membrane potential of HC
    events = nest.GetStatus(HCMMs[i])[0]['events']
    tPlot  = events['times'];
    plt.subplot(5, len(neuronsToRecord)+1, 0*(len(neuronsToRecord)+1)+i+1)
    plt.plot(tPlot, events['V_m'])
    plt.plot([0, simulationTime], [restPot, restPot], 'k-', lw=1)
    plt.axis([0, simulationTime, -90, -20])
    plt.ylabel('HC [mV]')

    # Plot the membrane potential of BC
    events = nest.GetStatus(BCMMs[i])[0]['events']
    tPlot  = events['times'];
    plt.subplot(5, len(neuronsToRecord)+1, 1*(len(neuronsToRecord)+1)+i+1)
    plt.plot(tPlot, events['V_m'])
    plt.plot([0, simulationTime], [restPot, restPot], 'k-', lw=1)
    plt.axis([0, simulationTime, -90, -20])
    plt.ylabel('BC [mV]')

    # Plot the membrane potential of AC
    events = nest.GetStatus(ACMMs[i])[0]['events']
    tPlot  = events['times'];
    plt.subplot(5, len(neuronsToRecord)+1, 2*(len(neuronsToRecord)+1)+i+1)
    plt.plot(tPlot, events['V_m'])
    plt.plot([0, simulationTime], [restPot, restPot], 'k-', lw=1)
    plt.axis([0, simulationTime, -90, -20])
    plt.ylabel('AC [mV]')

    # Plot the membrane potential of GC
    events = nest.GetStatus(GCMMs[i])[0]['events']
    tPlot  = events['times'];
    plt.subplot(5, len(neuronsToRecord)+1, 3*(len(neuronsToRecord)+1)+i+1)
    plt.plot(tPlot, events['V_m'])
    plt.plot([0, simulationTime], [threshPot, threshPot], 'k-', lw=1)
    plt.axis([0, simulationTime, -90, -20])
    plt.ylabel('GC [mV]')

    # Do the rasterplot
    plt.subplot(5, len(neuronsToRecord)+1, 4*(len(neuronsToRecord)+1)+i+1)
    plt.plot([startTime, stopTime], [1.25, 1.25], 'c-', lw=4)
    for spike in spikes:
        plt.plot([spike, spike], [0, 1], 'k-', lw=2)
    plt.axis([0, simulationTime, 0, 1.5])
    plt.ylabel('Rasterplot')

# Input shape
inputTime    = []
inputShapeHC = []
inputShapeBC = []
inputShapeAC = []
inputShapeGC = []
for time in range(timeSteps):
    inputTime.append(time)
    inputShapeHC.append(inputTimeFrame(RC_HC, 0.53, time, startTime, stopTime))
    inputShapeBC.append(inputTimeFrame(RC_BC, 0.61, time, startTime, stopTime))
    inputShapeAC.append(inputTimeFrame(RC_AC, 0.69, time, startTime, stopTime))
    inputShapeGC.append(inputTimeFrame(RC_GC, 1, time, startTime, stopTime))

plt.subplot(5,len(neuronsToRecord)+1, 1*(len(neuronsToRecord)+1))
plt.plot(inputTime, 1.0*numpy.array(inputShapeHC))
plt.subplot(5,len(neuronsToRecord)+1, 2*(len(neuronsToRecord)+1))
plt.plot(inputTime, 1.0*numpy.array(inputShapeBC))
plt.subplot(5,len(neuronsToRecord)+1, 3*(len(neuronsToRecord)+1))
plt.plot(inputTime, 1.0*numpy.array(inputShapeAC))
plt.subplot(5,len(neuronsToRecord)+1, 4*(len(neuronsToRecord)+1))
plt.plot(inputTime, 1.0*numpy.array(inputShapeGC))

# Show the plot
plt.show()
