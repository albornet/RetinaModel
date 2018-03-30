import nest, numpy, os, sys
import matplotlib.pyplot as plt
from makeGifs import gifMaker
nest.sli_run('M_WARNING setverbosity') # avoid writing too many NEST messages
nest.ResetKernel()
nest.SetKernelStatus({'resolution': 0.01, 'local_num_threads':4, 'print_time': True})


##########################
### Set the parameters ###
##########################

# Simulation parameters
simulationTime =  60.0      # [ms]
stepDuration   =   1.0      # [ms]  # put 1.0 here to see nice gifs
startTime      =   0.0      # [ms]
stopTime       =  20.0      # [ms]

# Retina parameters
BGCRatio       =    4
inhibRangeHC   =    2       # [pixels]
inhibRangeAC   =    2       # [pixels]
nRows          =   20       # [pixels]
nCols          =   20       # [pixels]

# Input parameters
inputTarget    =  (5, 5)    # [pixels]
inputRadius    =    5       # [pixels]
inputGain      = 1000.0     # [pA]
inputNoise     =   10.0     # [pA]

# Set the neurons whose LFP is going to be recorded
neuronsToRecord = [(inputTarget[0]+  0,           inputTarget[1]+0),
                   (inputTarget[0]+  1,           inputTarget[1]+0),
                   (inputTarget[0]+  inputRadius, inputTarget[1]+0),
                   (inputTarget[0]+2*inputRadius, inputTarget[1]+0)]

# Neurons custom parameters
threshPot      = -55.0
cellModel      = 'iaf_cond_alpha'
cellParams     = {'V_th': threshPot, 'tau_syn_ex': 1.0, 'V_reset': -70.0}

# Connection parameters
connections    = {
	'BC_To_GC' :  500.0*1,  # [nS/spike]
	'AC_To_GC' : -100.0*2,  # [nS/spike]
	'HC_To_BC' : -100.0*2,  # [nS/spike]
	'BC_To_AC' :   50.0*2}  # [nS/spike]

# Scale the weights, if needed
weightScale    = 1.0
for key, value in connections.items():
	connections[key] = value*weightScale


#########################
### Build the neurons ###
#########################

# Excitatory cells
GC = nest.Create(cellModel,          nRows*nCols, cellParams)
BC = nest.Create(cellModel, BGCRatio*nRows*nCols, cellParams)

# Inhibitory cells
nHCRows = int(float(nRows)/float(inhibRangeHC))+1
nHCCols = int(float(nCols)/float(inhibRangeHC))+1
nACRows = int(float(nRows)/float(inhibRangeAC))+1
nACCols = int(float(nCols)/float(inhibRangeAC))+1
AC = nest.Create(cellModel, nACRows*nACCols, cellParams)
HC = nest.Create(cellModel, nHCRows*nHCCols, cellParams)

# Spike detectors
GCSD = nest.Create('spike_detector',            nRows*  nCols)
BCSD = nest.Create('spike_detector', BGCRatio*  nRows*  nCols)
ACSD = nest.Create('spike_detector',          nACRows*nACCols)
HCSD = nest.Create('spike_detector',          nHCRows*nHCCols)

# Connect the spike detectors to their respective populations
nest.Connect(GC, GCSD, 'one_to_one')
nest.Connect(BC, BCSD, 'one_to_one')
nest.Connect(AC, ACSD, 'one_to_one')
nest.Connect(HC, HCSD, 'one_to_one')

# Create the gif makers, for each population
gifMakerList = []
gifMakerList.append(gifMaker(name='GC', popSD=GCSD, dimTuple=(1,            nRows,   nCols), orientedMap=False))
gifMakerList.append(gifMaker(name='BC', popSD=BCSD, dimTuple=(1, BGCRatio,  nRows,   nCols), orientedMap=True ))
gifMakerList.append(gifMaker(name='AC', popSD=ACSD, dimTuple=(1,          nACRows, nACCols), orientedMap=False))
gifMakerList.append(gifMaker(name='HC', popSD=HCSD, dimTuple=(1,          nHCRows, nHCCols), orientedMap=False))

# Create and connect the multimeter
GCMMs = []
for i in range(len(neuronsToRecord)):

	GCMM = nest.Create('multimeter', params = {'withtime': True, 'interval': 0.01, 'record_from': ['V_m', 'g_ex', 'g_in']})
	nest.Connect(GCMM, [GC[neuronsToRecord[i][0]*nCols + neuronsToRecord[i][1]]])
	GCMMs.append(GCMM)


#############################
### Build the connections ###
#############################

# Connections from bipolar cells to the retinal ganglion cells
source = []
target = []
for i in range(nRows):
	for j in range(nCols):
		for k in range(BGCRatio):
			source.append(k*nRows*nCols + i*nCols + j)
			target.append(                i*nCols + j)
nest.Connect([BC[s] for s in source], [GC[t] for t in target], 'one_to_one', {'weight': connections['BC_To_GC'], 'delay': 10.0})

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
						source.append(iAC       *nACCols + jAC         )
						target.append(convertedRow*nCols + convertedCol)
			nest.Connect([AC[s] for s in source], [GC[t] for t in target], 'one_to_one', {'weight': inhibACWeight, 'delay': 2.0})

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
							source.append(                iHC       *nHCCols + jHC         )
							target.append(k*nRows*nCols + convertedRow*nCols + convertedCol)
			nest.Connect([HC[s] for s in source], [BC[t] for t in target], 'one_to_one', {'weight': inhibHCWeight, 'delay': 2.0})

# Connections from bipolar cells to amacrine cells
source = []
target = []
inhibACWeight = connections['AC_To_GC']/numpy.sqrt(i**2 + j**2)
for iAC in range(nACRows):
	for jAC in range(nACCols):
		if 0 <= inhibRangeAC*iAC < nRows and 0 <= inhibRangeAC*jAC < nCols:
			for k in range(BGCRatio):
				source.append(k*nRows*nCols + inhibRangeAC*iAC*nCols   + inhibRangeAC*jAC)
				target.append(                             iAC*nACCols +              jAC)
nest.Connect([BC[s] for s in source], [AC[t] for t in target], 'one_to_one', {'weight': connections['BC_To_AC'], 'delay': 2.0})


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
						nest.SetStatus([BC[k*nRows*nCols + i*nRows + j]], {'I_e': inputStrength*0.4})

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
					nest.SetStatus([HC[iHC*nHCRows + jHC]], {'I_e': inputStrength*0.3})

	# Stop the stimulus
	if time == int(stopTime/stepDuration):
		nest.SetStatus(GC, {'I_e': 0.0})
		nest.SetStatus(BC, {'I_e': 0.0})
		nest.SetStatus(AC, {'I_e': 0.0})
		nest.SetStatus(HC, {'I_e': 0.0})

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
	events = nest.GetStatus(GCMMs[i])[0]['events']
	tPlot  = events['times'];

	# Do the rasterplot
	plt.subplot(3, len(neuronsToRecord), i+1)
	plt.plot([startTime, stopTime], [1.25, 1.25], 'c-', lw=4)
	for spike in spikes:
		plt.plot([spike, spike], [0, 1], 'k-', lw=2)
	plt.axis([0, simulationTime, 0, 1.5])
	plt.ylabel('Rasterplot')

	# Plot the membrane potential
	plt.subplot(3, len(neuronsToRecord), len(neuronsToRecord)+i+1)
	plt.plot(tPlot, events['V_m'])
	plt.plot([0, simulationTime], [threshPot, threshPot], 'k-', lw=1)
	plt.axis([0, simulationTime, -100, -25])
	plt.ylabel('Membrane potential [mV]')

	# Plot the conductance events
	plt.subplot(3, len(neuronsToRecord), 2*len(neuronsToRecord)+i+1)
	plt.plot(tPlot, events['g_ex'], tPlot, events['g_in'])
	plt.axis([0, simulationTime, 0, 5*max(numpy.abs(connections.values()))])
	plt.xlabel('Time [ms]')
	plt.ylabel('Synaptic conductance [nS]')
	plt.legend(('g_exc', 'g_inh'))

# Show the plot
plt.show()
