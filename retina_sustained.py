import nest, numpy, os, sys
import matplotlib.pyplot as plt
from makeGifs import gifMaker
from utilities import *

nCoresToUse = 4
nest.sli_run('M_WARNING setverbosity') # avoid writing too many NEST messages
nest.ResetKernel()
nest.SetKernelStatus({'resolution': 0.01, 'local_num_threads':nCoresToUse, 'print_time': True})


##########################
### Set the parameters ###
##########################

# Simulation parameters
simulationTime  =  100.0      # [ms]
stepDuration    =   1.0      # [ms]  # put 1.0 here to see nice gifs
startTime       =   0.0      # [ms]
stopTime        =  10.0      # [ms]
time            =  numpy.arange(0, simulationTime, stepDuration) # true time array, in [ms]
doDelta         =  True

# Retina parameters
nRows           = 10  # [pixels] --> how many cells (rows)
nCols           = 10  # [pixels] --> how many cells (cols)
BGCRatio        =  4
AGCRatio        =  5
HGCRatio        =  1
excitRangeBC    =  1
inhibRangeHC    =  1  # [pixels]
inhibRangeAC    =  2  # [pixels]
nonInhibRangeHC =  0  # [pixels]
nonInhibRangeAC =  1  # [pixels]

# Input parameters
inputTarget     =  (5, 5)            # [pixels]
inputRadius     =   4                # [pixels]
Voltage         = 200                # [mV]
inputVoltage    =   0.05*Voltage     # [mV]
inputNoise      =  (inputVoltage/10.0)*0.0
shape           =  'prosthetic'

# Layers z-position
z_GC            = 10 # [um]
z_AC            = 30 # [um]
z_BC            = 49  # [um]
z_HC            = 64  # [um]

# Get the low-pass filter time constant associated to each layer membrane and position
RC_GC           = getRC(z_GC, 10)*10**3  # 0.4 [ms]
RC_AC           = getRC(z_AC,  5)*10**3  # 0.65[ms]
RC_BC           = getRC(z_BC,  5)*10**3  # 10  [ms]
RC_HC           = getRC(z_HC, 10)*10**3  # 12  [ms]

# Get the delays associated to each layer z-position
delayGC         = getDelay(z_GC, Voltage)
delayAC         = getDelay(z_AC, Voltage)
delayBC         = getDelay(z_BC, Voltage)
delayHC         = getDelay(z_HC, Voltage)

# Set the input for each neuron, taking into account the attenuation factor (depends on z, but was measured)
input_GC        = inputTimeFrame(RC_GC, 0.90*inputVoltage, inputNoise, time, startTime + delayGC, stopTime + delayGC, shape)
input_AC        = inputTimeFrame(RC_AC, 0.42*inputVoltage, inputNoise, time, startTime + delayAC, stopTime + delayAC, shape)
input_BC        = inputTimeFrame(RC_BC, 0.31*inputVoltage, inputNoise, time, startTime + delayBC, stopTime + delayBC, shape)
input_HC        = inputTimeFrame(RC_HC, 0.25*inputVoltage, inputNoise, time, startTime + delayHC, stopTime + delayHC, shape)

# Set the neurons whose LFP is going to be recorded
neuronsToRecord = [(inputTarget[0]+  0,           inputTarget[1]+0),
                   (inputTarget[0]+  1,           inputTarget[1]+0),
                   (inputTarget[0]+  inputRadius, inputTarget[1]+0)]
                   # (inputTarget[0]+2*inputRadius, inputTarget[1]+0)]

# Neurons custom parameters
threshPot         = -55.0
restPot           = -70.0  # more or less equal for all populations taking into account std in litterature
neuronModel       = 'aeif_cond_alpha'
interneuronModel  = 'iaf_cond_alpha'

# neuronParams      = {'V_th': threshPot, 't_ref': 0.0, 'V_reset': -47.0, 'a': 0.0, 'b': 10.0, 'tau_w': 144.0   'E_L': restPot}
# neuronParams      = {'V_th':  -52.0, 't_ref': 0.0, 'V_reset': -47.0, 'a': 0.0, 'b': 80.0, 'tau_w': 50.0,   'E_L': restPot}
# neuronParams      = {'V_th':  -55.0, 't_ref': 0.0, 'V_reset': -54.0, 'a': 0.0, 'b':-50.0, 'tau_w': 30.0, 'E_L': restPot}
neuronParams      = {'V_th': threshPot, 't_ref': 2.0, 'V_reset': -54.0, 'a': -100.0, 'b': 1.0,  'tau_w': 500.0, 'E_L': restPot} # avec AC_To_BC = -1000
interNeuronParams = {'V_th': 1000.0,    't_ref': 3.5, 'V_reset': -70.0, 'tau_syn_ex': 1.0, 'tau_syn_in':  1.0, 'E_L': restPot}

# Connection parameters
connections    = {
    'BC_To_GC' :  700, #  700 [nS/spike]
    'AC_To_GC' : -700, # -700 [nS/spike]
    'HC_To_BC' :  -50, #  -50 [nS/spike]
    'BC_To_AC' :   70, #   70 [nS/spike]
    'GC_auto'  :    0
    }

# Scale the weights, if needed
weightScale    = 0.0002    # 0.0005
for key, value in connections.items():
    connections[key] = value*weightScale


#########################
### Build the neurons ###
#########################

# Cells
GC = nest.Create(neuronModel,               nRows*nCols,      neuronParams)
BC = nest.Create(interneuronModel, BGCRatio*nRows*nCols, interNeuronParams)
AC = nest.Create(interneuronModel, AGCRatio*nRows*nCols, interNeuronParams)
HC = nest.Create(interneuronModel, HGCRatio*nRows*nCols, interNeuronParams)

# Previous membrane potential (previous time-step) ; initialized at resting pot.
BCLastVoltage = numpy.zeros((len(BC),))
ACLastVoltage = numpy.zeros((len(AC),))
HCLastVoltage = numpy.zeros((len(HC),))

# Spike detectors
GCSD = nest.Create('spike_detector', nRows* nCols)

# Connect the spike detectors to their respective populations
nest.Connect(GC, GCSD, 'one_to_one', {"weight": connections['GC_auto']})

# # Create the gif makers, for each population
# gifMakerList = []
# gifMakerList.append(gifMaker(name='GC', popID=GCSD, dimTuple=(1,            nRows,   nCols), orientedMap=False, spiking=True , baseline=None   ))
# # gifMakerList.append(gifMaker(name='BC', popID=BC,   dimTuple=(1, BGCRatio,  nRows,   nCols), orientedMap=True , spiking=False, baseline=restPot))
# # gifMakerList.append(gifMaker(name='AC', popID=AC,   dimTuple=(1,          nACRows, nACCols), orientedMap=False, spiking=False, baseline=restPot))
# # gifMakerList.append(gifMaker(name='HC', popID=HC,   dimTuple=(1,          nHCRows, nHCCols), orientedMap=False, spiking=False, baseline=restPot))


# # Create and connect the multimeter to simulate interneurons
# BCMMs = []
# ACMMs = []
# HCMMs = []
#
# # Bipolar cells multimeters (vesicles fusion proportionnal to their dpotential/dt)
# for i in range(len(BCSD)):
#
#   BCMM = nest.Create('multimeter', params = {'withtime': True, 'interval': stepDuration, 'record_from': ['V_m']})
#   nest.Connect(BCMM, [BC[i]])
#   BCMMs.append(BCMM)
#
# # Amacrine cells multimeters (vesicles fusion proportionnal to their dpotential/dt)
# for i in range(len(neuronsToRecord)):
#
#   ACMM = nest.Create('multimeter', params = {'withtime': True, 'interval': stepDuration, 'record_from': ['V_m']})
#   nest.Connect(ACMM, [AC[i]])
#   ACMMs.append(ACMM)
#
# # Horizontal cells multimeters (vesicles fusion proportionnal to their dpotential/dt)
# for i in range(len(neuronsToRecord)):
#
#   HCMM = nest.Create('multimeter', params = {'withtime': True, 'interval': stepDuration, 'record_from': ['V_m']})
#   nest.Connect(HCMM, [HC[i]])
#   HCMMs.append(HCMM)

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
for t in range(timeSteps):

    # Ganglion cells input
    for i in range(nRows):
        for j in range(nCols):

            distance = numpy.sqrt((i-inputTarget[0])**2 + (j-inputTarget[1])**2)
            if distance < inputRadius:

                StimGC    = input_GC[t]*inputSpaceFrame(distance, 0.5*inputRadius)
                target    = (i*nCols + j)
                GCVoltage = nest.GetStatus([GC[target]], 'V_m')[0] - restPot
                nest.SetStatus([GC[target]], {'V_m': restPot + GCVoltage + StimGC})

    # Amacrine cells input
    for i in range(nRows):
        for j in range(nCols):

            distance = numpy.sqrt((i-inputTarget[0])**2 + (j-inputTarget[1])**2)
            if distance < inputRadius:

                StimAC = input_AC[t]*inputSpaceFrame(distance, 0.5*inputRadius)
                for k in range(AGCRatio):

                    target    = (k*nRows*nCols + i*nRows + j)
                    ACVoltage = nest.GetStatus([AC[target]], 'V_m')[0] - restPot
                    nest.SetStatus([AC[target]], {'V_m': restPot + ACVoltage + StimAC})

    # Bipolar cells input
    for i in range(nRows):
        for j in range(nCols):
            distance = numpy.sqrt((i-inputTarget[0])**2 + (j-inputTarget[1])**2)
            if distance < inputRadius:

                StimBC = input_BC[t]*inputSpaceFrame(distance, 0.5*inputRadius)
                for k in range(BGCRatio):

                    target    = (k*nRows*nCols + i*nRows + j)
                    BCVoltage = nest.GetStatus([BC[target]], 'V_m')[0]- restPot
                    nest.SetStatus([BC[target]], {'V_m': restPot + BCVoltage + StimBC})

    # Horizontal cells input
    for i in range(nRows):
        for j in range(nCols):
            distance = numpy.sqrt((i-inputTarget[0])**2 + (j-inputTarget[1])**2)
            if distance < inputRadius:

                StimHC = input_HC[t]*inputSpaceFrame(distance, 0.5*inputRadius)
                for k in range(HGCRatio):

                    target    = (k*nRows*nCols + i*nRows + j)
                    HCVoltage = nest.GetStatus([HC[target]], 'V_m')[0] - restPot
                    nest.SetStatus([HC[target]], {'V_m': restPot + HCVoltage + StimHC})

    # Connections from bipolar cells to the retinal ganglion cells
    source = []
    target = []
    for i in range(nRows):
        for j in range(nCols):
            for kBC in range(BGCRatio):

                source = (kBC*nRows*nCols + i*nCols + j)
                target = (                  i*nCols + j)
                preSynVoltage         = nest.GetStatus([BC[source]], 'V_m')[0] - restPot
                deltaPreSynVoltage    = (preSynVoltage - BCLastVoltage[source])/stepDuration
                if (deltaPreSynVoltage > 0.0 or not doDelta):
                    postSynVoltage    = nest.GetStatus([GC[target]], 'V_m')[0] - restPot
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

                                source = (kAC*nRows*nCols + i    *nCols +  j    )
                                target = (                 (i+i2)*nCols + (j+j2))
                                preSynVoltage  = nest.GetStatus([AC[source]], 'V_m')[0] - restPot
                                deltaPreSynVoltage    = (preSynVoltage - ACLastVoltage[source])/stepDuration
                                if (deltaPreSynVoltage > 0.0 or not doDelta):
                                    postSynVoltage    = nest.GetStatus([GC[target]], 'V_m')[0] - restPot
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
                                    deltaPreSynVoltage    = (preSynVoltage - HCLastVoltage[source])/stepDuration
                                    if (deltaPreSynVoltage > 0.0 or not doDelta):
                                        postSynVoltage    = nest.GetStatus([BC[target]], 'V_m')[0] - restPot
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
                                    deltaPreSynVoltage    = (preSynVoltage - BCLastVoltage[source])/stepDuration
                                    if (deltaPreSynVoltage > 0.0 or not doDelta):
                                        postSynVoltage    = nest.GetStatus([AC[target]], 'V_m')[0] - restPot
                                        nest.SetStatus([AC[target]], {'V_m': restPot + postSynVoltage + connections['BC_To_AC']*preSynVoltage})
    # autoconnection GC
    nest.Connect(GC, GC, 'one_to_one',)

    # Update the last time-step presynaptic voltages
    for i in range(nRows):
        for j in range(nCols):
            for k in range(BGCRatio):
                source = k*nRows*nCols + i*nCols + j
                BCLastVoltage[source] = nest.GetStatus([BC[source]], 'V_m')[0] - restPot
            for k in range(AGCRatio):
                source = k*nRows*nCols + i*nCols + j
                ACLastVoltage[source] = nest.GetStatus([AC[source]], 'V_m')[0] - restPot
            for k in range(HGCRatio):
                source = k*nRows*nCols + i*nCols + j
                HCLastVoltage[source] = nest.GetStatus([HC[source]], 'V_m')[0] - restPot

    # Run the simulation for one gif frame
    try:
        nest.Simulate(stepDuration)
    except:
        continue
    if t < timeSteps-1:
        sys.stdout.write("\033[2F") # move the cursor back to previous line

    # # Take screenshots of every recorded population
    # if t%25 == 0:
    #     for instance in gifMakerList: # gifMaker.getInstances():
    #         (namePop, nSpikes) = instance.takeScreenshot()


#################################
### Read the network's output ###
#################################

# # Create animated gif of stimulus
# sys.stdout.write('Creating animated gifs.\n\n')
# sys.stdout.flush()
# for instance in gifMakerList: # gifMaker.getInstances():
#     instance.createGif(figureDir, durationTime=0.2)

f = open('SimFigures/Spikes.txt', 'w')
centralNeurons       = [0,1]
centralNeuronsSpikes = []
plt.figure()
for i in range(len(neuronsToRecord)):

    # Obtain and display data
    recRow = neuronsToRecord[i][0]
    recCol = neuronsToRecord[i][1]
    spikes = nest.GetStatus([GCSD[recRow*nRows+recCol]], keys='events')[0]['times']
    Spikes = numpy.asarray(spikes)
    SL     = numpy.sum(numpy.array([x for x in Spikes if x<10]))/(10*0.001)                  # [Hz]
    ML     = numpy.sum(numpy.array([x for x in Spikes if x>40 and x<120]))/((120-40)*0.001)  # [Hz]
    #print(SL)
    #print(ML)
    print(spikes, len(spikes))
    f.write('\n'+'Spikes times of neuron '+str(i)+': ')
    for spike in spikes:
        f.write(str(spike)+'\t')
        if i in centralNeurons:
            centralNeuronsSpikes.append(spike)

    # Plot the membrane potential of HC
    events = nest.GetStatus(HCMMs[i])[0]['events']
    tPlot  = events['times'];
    plt.subplot(5, len(neuronsToRecord)+1, 0*(len(neuronsToRecord)+1)+i+1)
    plt.plot(tPlot, events['V_m'])
    plt.plot([0, simulationTime], [restPot, restPot], 'k-', lw=1)
    plt.axis([0, simulationTime, min(events['V_m'])-5, max(events['V_m']+5)])
    plt.ylabel('HC [mV]')

    # Plot the membrane potential of BC
    events = nest.GetStatus(BCMMs[i])[0]['events']
    tPlot  = events['times'];
    plt.subplot(5, len(neuronsToRecord)+1, 1*(len(neuronsToRecord)+1)+i+1)
    plt.plot(tPlot, events['V_m'])
    plt.plot([0, simulationTime], [restPot, restPot], 'k-', lw=1)
    plt.axis([0, simulationTime, min(events['V_m'])-5, max(events['V_m']+5)])
    plt.ylabel('BC [mV]')

    # Plot the membrane potential of AC
    events = nest.GetStatus(ACMMs[i])[0]['events']
    tPlot  = events['times'];
    plt.subplot(5, len(neuronsToRecord)+1, 2*(len(neuronsToRecord)+1)+i+1)
    plt.plot(tPlot, events['V_m'])
    plt.plot([0, simulationTime], [restPot, restPot], 'k-', lw=1)
    plt.axis([0, simulationTime, min(events['V_m'])-5, max(events['V_m']+5)])
    plt.ylabel('AC [mV]')

    # Plot the membrane potential of GC
    events = nest.GetStatus(GCMMs[i])[0]['events']
    tPlot  = events['times'];
    plt.subplot(5, len(neuronsToRecord)+1, 3*(len(neuronsToRecord)+1)+i+1)
    plt.plot(tPlot, events['V_m'])
    plt.plot([0, simulationTime], [threshPot, threshPot], 'k-', lw=1)
    plt.axis([0, simulationTime, min(events['V_m'])-5, max(events['V_m']+5)])
    plt.ylabel('GC [mV]')

    # Do the rasterplot
    plt.subplot(5, len(neuronsToRecord)+1, 4*(len(neuronsToRecord)+1)+i+1)
    plt.plot([startTime, stopTime], [1.25, 1.25], 'c-', lw=4)
    for spike in spikes:
        plt.plot([spike, spike], [0, 1], 'k-', lw=2)
    plt.axis([0, simulationTime, 0, 1.5])
    plt.ylabel('Rasterplot')

# Close the spike file and do the spikes histogram
f.close()
plt.subplot(5, len(neuronsToRecord)+1, 5*(len(neuronsToRecord)+1))
plt.hist(x=centralNeuronsSpikes, bins=int(simulationTime/10.0), range=(0,simulationTime), weights=[1.0/len(centralNeurons) for i in centralNeuronsSpikes])

# Plot different inputs (for each layer)
plt.subplot(5,len(neuronsToRecord)+1, 1*(len(neuronsToRecord)+1))
plt.plot(time, input_HC)
plt.subplot(5,len(neuronsToRecord)+1, 2*(len(neuronsToRecord)+1))
plt.plot(time, input_BC)
plt.subplot(5,len(neuronsToRecord)+1, 3*(len(neuronsToRecord)+1))
plt.plot(time, input_AC)
plt.subplot(5,len(neuronsToRecord)+1, 4*(len(neuronsToRecord)+1))
plt.plot(time, input_GC)


# Show and save the plot
# plt.savefig('SimFigures/Raster.eps', format='eps', dpi=1000)
plt.savefig('./Results_Sustain/coucou.eps', format='eps', dpi=1000)
plt.show()