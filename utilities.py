import numpy

def getRC(d, D):

    A=  300000  # [um2] area of one neuron
    return (10*d*10**-6 + (5*10**8)*A*10**-12*(300/7)*(d/7)) * ((0.1*d*10**-6)**-1 + (0.01*A*10**-12*(300/7)*(d/7))**-1)**-1


# Calculate ions mobility delay
def getDelay(distance, voltage):                                                     # [um][mV]

    #return ((distance*10**-6)**2/((voltage*10**-3)*363*10**-9))*10**3               # [ms] if charge carrier speed cst
    d0 = (-6*10**-8*(voltage**3))+(0.0002*(voltage**2))+(0.1471*voltage)-6.2835      # [um]
    return 0 #((((d0*10**-6)**2)/(363*10**-9*voltage*10**-3))*(numpy.exp((distance*10**-6)/(d0*10**-6))-1))


def voltagePattern(V_gain, time, t_start, t_stop, shape):

    if t_start < time < t_stop:
        #if shape == 'square':
        #    return V_gain
        if shape == 'prosthetic':
            return V_gain*(1-numpy.exp(-(time-t_start)/2.0))
        #if shape == 'triangle':
        	#return V_gain/(t_stop-t_start)*(time-t_start)
    if time >= t_stop:
    	#if shape == 'square':
    	#	return 0.0
        if shape == 'prosthetic':
        	return V_gain*(1-numpy.exp(-(t_stop-t_start)/2.0))*numpy.exp(-(time-t_stop)/12.0)
        #if shape == 'triangle':
        	#return max(-V_gain/(t_stop-t_start)*(time-(t_start+t_stop)), 0.0)
    else:
        return 0.0


def inputSpaceFrame(d, sigma):

    return numpy.exp(-d**2/(2*sigma**2))


def inputTimeFrame(RC, V_gain, input_noise, t, t_start, t_stop, shape):

    neuronFilter = numpy.exp(-t/(RC))
    neuronFilter = neuronFilter/numpy.sum(neuronFilter)
    inputVoltage = numpy.zeros(t.shape)
    for i, t_ in enumerate(t):

        noise_gain      = input_noise*(numpy.random.rand()-0.5)*2.0
        V_strength      = V_gain + noise_gain
        inputVoltage[i] = voltagePattern(V_strength, t_, t_start, t_stop, shape)

    return numpy.convolve(inputVoltage, neuronFilter, 'full')[0:len(t)]
