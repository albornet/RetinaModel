import nest, numpy, os, sys
import matplotlib.pyplot as plt
from makeGifs import gifMaker
nCoresToUse = 1
nest.sli_run('M_WARNING setverbosity') # avoid writing too many NEST messages
nest.ResetKernel()
nest.SetKernelStatus({'resolution': 0.01, 'local_num_threads':nCoresToUse, 'print_time': True})

# To do:
# SAVE spikestimes + SL/ML/LL

##########################
### Set the parameters ###
##########################

# Simulation parameters
simulationTime =  20.0      # [ms]
stepDuration   =   1.0      # [ms]  # put 1.0 here to see nice gifs
startTime      =   0.0      # [ms]
stopTime       =  50.0      # [ms]

# Retina parameters
BGCRatio        =    4
AGCRatio        =    5
HGCRatio        =    1
excitRangeBC    =    1
inhibRangeHC    =    1            # [pixels]
inhibRangeAC    =    2            # [pixels]
nonInhibRangeHC =    0            # [pixels]
nonInhibRangeAC =    1            # [pixels]

def getRC(d, D):                         #[um]
    #return (0.7*d*10**-6 + 0.7*4*numpy.pi*(D*10**-6)**2)     * ((0.1*d*10**-6)**-1 + (0.01*4*numpy.pi*(D*10**-6)**2)**-1)**-1
    r= 40000000000000                                #[um]
    return  (0.7*d*10**-6 + 0.7*2*numpy.pi*r*10**-6*d*10**-6) * ((0.1*d*10**-6)**-1 + (0.01*2*numpy.pi*r*10**-6*d*10**-6)**-1)**-1

RC_GC           =   getRC(10,10)*10**3   # 0.4 [ms]
RC_BC           =   getRC(49,5)*10**3    # 10  [ms]
RC_AC           =   getRC(30,5)*10**3    # 0.65[ms]
RC_HC           =   getRC(64,10)*10**3   # 12  [ms]
nRows           =   10                   # [pixels]
nCols           =   10                   # [pixels]

# Input parameters
inputTarget    =   (5, 5)            # [pixels]
inputRadius    =    4                # [pixels]
Voltage        =   150               # [mV]
inputVoltage   =   0.05*Voltage      # [mV]
inputNoise     =   10.0
def inputSpaceFrame(d, sigma):
    return numpy.exp(-d**2/(2*sigma**2))
def inputTimeFrame(RC, gain, t, start, stop):
    if start < t < stop:
        return gain*(1-numpy.exp(-(t-start)/RC))
    if t >= stop:
        return gain*(1-numpy.exp(-(stop-start)/RC))*numpy.exp (-(t-stop)/RC)
    else:
        return 0.0

# Set the neurons whose LFP is going to be recorded
neuronsToRecord = [(inputTarget[0]+  0,           inputTarget[1]+0),
                   (inputTarget[0]+  1,           inputTarget[1]+0),
                   (inputTarget[0]+  inputRadius, inputTarget[1]+0)]
                   # (inputTarget[0]+2*inputRadius, inputTarget[1]+0)]

# Neurons custom parameters
threshPot         = -55.0
restPot           = -70.0  # more or less equal for all populations taking into account std in litterature
neuronModel       = 'iaf_cond_alpha'
neuronParams      = {'V_th': threshPot,      'tau_syn_ex': 10.0, 'tau_syn_in': 10.0, 'V_reset': -70.0, 't_ref': 3.5}
interNeuronParams = {'V_th': threshPot+1000, 'tau_syn_ex': 1.0,  'tau_syn_in':  1.0, 'V_reset': -70.0, 't_ref': 3.5}

# Connection parameters
connections    = {
    'BC_To_GC' : 700, #  7000 [nS/spike]
    'AC_To_GC' :-700, # -7000 [nS/spike]
    'HC_To_BC' : -70, #  -700 [nS/spike]
    'BC_To_AC' :  70  #   700 [nS/spike]
    }

# Scale the weights, if needed
weightScale    = 0.0002    # 0.0005
for key, value in connections.items():
    connections[key] = value*weightScale


#########################
### Build the neurons ###
#########################

# Cells
GC = nest.Create(neuronModel,          nRows*nCols,      neuronParams)
BC = nest.Create(neuronModel, BGCRatio*nRows*nCols, interNeuronParams)
AC = nest.Create(neuronModel, AGCRatio*nRows*nCols, interNeuronParams)
HC = nest.Create(neuronModel, HGCRatio*nRows*nCols, interNeuronParams)

# Previous membrane potential (previous time-step) ; initialized at resting pot.
BCLastVoltage = numpy.zeros((len(BC),))
ACLastVoltage = numpy.zeros((len(AC),))
HCLastVoltage = numpy.zeros((len(HC),))

# Spike detectors
GCSD = nest.Create('spike_detector', nRows* nCols)

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
# # Bipolar cells multimeters (vesicles fusion proportionnal to their dpotential/dt)
# for i in range(len(BCSD)):
#
# 	BCMM = nest.Create('multimeter', params = {'withtime': True, 'interval': stepDuration, 'record_from': ['V_m']})
# 	nest.Connect(BCMM, [BC[i]])
# 	BCMMs.append(BCMM)
#
# # Amacrine cells multimeters (vesicles fusion proportionnal to their dpotential/dt)
# for i in range(len(neuronsToRecord)):
#
# 	ACMM = nest.Create('multimeter', params = {'withtime': True, 'interval': stepDuration, 'record_from': ['V_m']})
# 	nest.Connect(ACMM, [AC[i]])
# 	ACMMs.append(ACMM)
#
# # Horizontal cells multimeters (vesicles fusion proportionnal to their dpotential/dt)
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

# Calculate ions mobility delay
def d0(Voltage):                                                                     #[mV]
    return (-6*10**-8*(Voltage**3))+(0.0002*(Voltage**2))+(0.1471*Voltage)-6.2835    #[um]
def delay(distance,voltage):                                                         # [um][mV]
    #return ((distance*10**-6)**2/((voltage*10**-3)*363*10**-9))*10**3               # [ms] if charge carrier speed cst
    return ((((d0(voltage)*10**-6)**2)/(363*10**-9*voltage*10**-3))*(numpy.exp((distance*10**-6)/(d0(voltage)*10**-6))-1))
delayGC = delay(10,Voltage)
delayAC = delay(30,Voltage)
delayBC = delay(49,Voltage)
delayHC = delay(64,Voltage)

# Simulate the network
timeSteps = int(simulationTime/stepDuration)
for time in range(timeSteps):

    # Ganglion cells input
    for i in range(nRows):
        for j in range(nCols):
            distance = numpy.sqrt((i-inputTarget[0])**2 + (j-inputTarget[1])**2)
            if distance < inputRadius:
                noiseGain     = inputNoise*(numpy.random.rand()-0.5)*2.0
                inputStrength = (inputVoltage+noiseGain)*inputSpaceFrame(distance, 0.5*inputRadius)
                StimGC= inputTimeFrame(RC_GC, inputStrength, time, startTime + delayGC, stopTime + delayGC)
                target        = (                 i*nCols + j)
                GCVoltage = nest.GetStatus([GC[target]], 'V_m')[0] - restPot
                nest.SetStatus([GC[target]], {'V_m': restPot + GCVoltage + StimGC*0.9})

    # Amacrine cells input
    for i in range(nRows):
        for j in range(nCols):
            distance = numpy.sqrt((i-inputTarget[0])**2 + (j-inputTarget[1])**2)
            if distance < inputRadius:
                noiseGain     = inputNoise*(numpy.random.rand()-0.5)*2.0
                inputStrength = (inputVoltage+noiseGain)*inputSpaceFrame(distance, 0.5*inputRadius)
                StimAC= inputTimeFrame(RC_AC, inputStrength, time, startTime + delayAC, stopTime + delayAC)
                for k in range(AGCRatio):
                    target    = (k*nRows*nCols + i*nRows + j)
                    ACVoltage = nest.GetStatus([AC[target]], 'V_m')[0] - restPot
                    nest.SetStatus([AC[target]], {'V_m': restPot + ACVoltage + StimAC*0.42})

    # Bipolar cells input
    for i in range(nRows):
        for j in range(nCols):
            distance = numpy.sqrt((i-inputTarget[0])**2 + (j-inputTarget[1])**2)
            if distance < inputRadius:
                noiseGain     = inputNoise*(numpy.random.rand()-0.5)*2.0
                inputStrength = (inputVoltage+noiseGain)*inputSpaceFrame(distance, 0.5*inputRadius)
                StimBC= inputTimeFrame(RC_BC, inputStrength, time, startTime + delayBC, stopTime + delayBC)
                for k in range(BGCRatio):
                    target    = (k*nRows*nCols + i*nRows + j)
                    BCVoltage = nest.GetStatus([BC[target]], 'V_m')[0]- restPot
                    nest.SetStatus([BC[target]], {'V_m': restPot + BCVoltage + StimBC*0.31})

    # Horizontal cells input
    for i in range(nRows):
        for j in range(nCols):
            distance = numpy.sqrt((i-inputTarget[0])**2 + (j-inputTarget[1])**2)
            if distance < inputRadius:
                noiseGain     = inputNoise*(numpy.random.rand()-0.5)*2.0
                inputStrength = (inputVoltage+noiseGain)*inputSpaceFrame(distance, 0.5*inputRadius)
                StimHC= inputTimeFrame(RC_HC, inputStrength, time, startTime + delayHC, stopTime + delayHC)
                for k in range(HGCRatio):
                    target    = (k*nRows*nCols + i*nRows + j)
                    HCVoltage = nest.GetStatus([HC[target]], 'V_m')[0] - restPot
                    nest.SetStatus([HC[target]], {'V_m': restPot + HCVoltage + StimHC*0.25})

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
                if deltaPreSynVoltage > 0.0:
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
                                source = (kAC*nRows*nCols + i     *nCols +  j    )
                                target = (                 (i+i2)*nCols + (j+j2))
                                preSynVoltage  = nest.GetStatus([AC[source]], 'V_m')[0] - restPot
                                deltaPreSynVoltage    = (preSynVoltage - ACLastVoltage[source])/stepDuration
                                if deltaPreSynVoltage > 0.0:
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
                                    if deltaPreSynVoltage > 0.0:
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
                                    if deltaPreSynVoltage > 0.0:
                                        postSynVoltage    = nest.GetStatus([AC[target]], 'V_m')[0] - restPot
                                        nest.SetStatus([AC[target]], {'V_m': restPot + postSynVoltage + connections['BC_To_AC']*preSynVoltage})

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

f = open('SimFigures/Spikes.txt', 'w')
centralNeurons       = [0,1]
centralNeuronsSpikes = []
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

# Input shape
inputTime    = []
inputShapeHC = []
inputShapeBC = []
inputShapeAC = []
inputShapeGC = []
for time in range(timeSteps):
    inputTime.append(time)
    inputShapeHC.append(inputTimeFrame(RC_HC, 0.25, time, startTime + delayHC, stopTime + delayHC))
    inputShapeBC.append(inputTimeFrame(RC_BC, 0.31, time, startTime + delayBC, stopTime + delayBC))
    inputShapeAC.append(inputTimeFrame(RC_AC, 0.42, time, startTime + delayAC, stopTime + delayAC))
    inputShapeGC.append(inputTimeFrame(RC_GC, 0.90, time, startTime + delayGC, stopTime + delayGC))

plt.subplot(5,len(neuronsToRecord)+1, 1*(len(neuronsToRecord)+1))
plt.plot(inputTime, 1.0*numpy.array(inputShapeHC))
plt.subplot(5,len(neuronsToRecord)+1, 2*(len(neuronsToRecord)+1))
plt.plot(inputTime, 1.0*numpy.array(inputShapeBC))
plt.subplot(5,len(neuronsToRecord)+1, 3*(len(neuronsToRecord)+1))
plt.plot(inputTime, 1.0*numpy.array(inputShapeAC))
plt.subplot(5,len(neuronsToRecord)+1, 4*(len(neuronsToRecord)+1))
plt.plot(inputTime, 1.0*numpy.array(inputShapeGC))

# Show % save the plot
plt.savefig('SimFigures/Raster.eps', format='eps', dpi=1000)
plt.show()
