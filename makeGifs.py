# Imports
import nest, numpy
from images2gif import writeGif

# Class to draw gifs out of population activity snapshots ; one per population
class gifMaker:

	# Initialize the variables of the class
	def __init__(self, name, popID, dimTuple, orientedMap, spiking, baseline):

		self.name        = name                     # name of the population
		self.popID       = popID                    # spike detector for the population OR state of the population ('V_m')
		self.dimTuple    = dimTuple                 # dimensions of the population (e.g. (nOri, nRows, nCols))
		self.plotVector  = numpy.zeros(len(popID))  # snapshot of the population activity
		self.cumuVector  = numpy.zeros(len(popID))  # cumulation of the activity to get the snapshots
		self.maxAllTime  = 1.0                      # normalization of the gifs
		self.outImages   = [[] for i in range(dimTuple[0])]  # what will be plotted
		self.orientedMap = orientedMap              # boolean (will the gif contain orientations=colors or not)
		self.spiking     = spiking                  # boolean (true if we gif the spikes, false if we gif V_m
		self.baseline    = baseline                 # resting potential (used to plot positive V_m values)

	# Screen the activity of the population at a certain time-step
	def takeScreenshot(self):

		# Record the spikes of the current step and update the cumulative vector (to substract previous spikes)
		if self.spiking:
			self.plotVector  = nest.GetStatus(self.popID, 'n_events') - self.cumuVector
			self.cumuVector += self.plotVector
		else:
			self.plotVector  = [voltage - self.baseline for voltage in nest.GetStatus(self.popID, 'V_m')]

		# Take care of output normalization
		maxThisTime      = numpy.max(self.plotVector)
		if 1: # maxThisTime > 0:
			self.maxAllTime = max(maxThisTime, self.maxAllTime)

			# Oriented output screenshot
			if self.orientedMap:

				(nSeg, nOri, nRow, nCol) = self.dimTuple
				thisPlot = numpy.reshape(self.plotVector, self.dimTuple)
				for h in range(nSeg):
					if nOri == 2:
						self.outImages[h].append(numpy.dstack((thisPlot[h,0,:,:], thisPlot[h,1,:,:], numpy.zeros((nRow,nCol)))))
					if nOri == 4:
						self.outImages[h].append(numpy.dstack((thisPlot[h,1,:,:], thisPlot[h,3,:,:], numpy.maximum(thisPlot[h,0,:,:], thisPlot[h,2,:,:]))))
					if nOri == 8:
						rgbMap = numpy.array([[0.,.5,.5], [0.,0.,1.], [.5,0.,.5], [1.,0.,0.], [.5,0.,.5], [0.,0.,1.], [0.,.5,.5], [0.,1.,0.]])
						self.outImages[h].append(numpy.tensordot(thisPlot[h,:,:,:], rgbMap, axes=(0,0)))

			# Simple output screenshot
			else:
				(nSeg, nRow, nCol) = self.dimTuple
				for h in range(nSeg):
					thisImage = numpy.reshape(self.plotVector, self.dimTuple)[h,:,:]
					self.outImages[h].append(numpy.dstack((thisImage, numpy.zeros((nRow, nCol)), numpy.zeros((nRow, nCol)))))

		# Useful for the retina code
		return (self.name, numpy.sum(self.plotVector))

	# Generate a gif out of the snapshots that were taken, after the network was simulated
	def createGif(self, thisTrialDir, durationTime):

		# Create an animated gif of the output ; rescale firing rates to max value
		if self.dimTuple[0] > 1:
			for h in range(self.dimTuple[0]):
				writeGif(thisTrialDir+'/'+self.name+'Seg'+str(h)+'.gif', numpy.array(self.outImages[h])/self.maxAllTime, duration=durationTime)
		else:
			writeGif(thisTrialDir+'/'+self.name+'.gif', numpy.array(self.outImages[0])/self.maxAllTime, duration=durationTime)

		# Reset the output images, for the next trials
		self.outImages = [[] for i in range(self.dimTuple[0])]
