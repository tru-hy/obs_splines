import numpy as np
import matplotlib.pyplot as plt

from obs_splines import *

def test_spline():
	# Create a test signal
	x, y = blocked_sine()
	
	# Add some noise (actually no noise here, makes for
	# cleaner plots)
	noise = np.random.randn(len(y))*0.0
	
	# Add some spiky noise
	n_spikes = 10
	spikes = np.random.choice(len(y), n_spikes)
	noise[spikes] += (np.random.rand(n_spikes)-0.5)*5
	sig = y + noise
	
	# Smoothing criteria, larger is smoother.
	# Perhaps some day the documentation will explain
	# a bit more.
	smoothing = 3.0
	plt.plot(x, y, label='truth')
	plt.plot(x, sig, label='observations')
	
	# Smooth using (robust) L1/Laplacean error model
	smoothedl1 = smooth1d_grid_l1_l2_missing(sig, smoothing=smoothing)
	plt.plot(x, smoothedl1, label='L1 smoothed')
	
	# Smooth using (non-robust but faster) L2/Gaussian error model
	smoothedl2 = smooth1d_grid_l2_l2_simple(sig, smoothing=smoothing)
	plt.plot(x, smoothedl2, label='L2 smoothed')
	plt.legend()

	plt.show()


def blocked_sine(dur=10, dx=0.01):
	x = np.arange(0, dur, dx)
	y = np.sin(x)
	midpoint = len(x)/2
	y[midpoint-1/dx:midpoint+1/dx] = 0.0
	return x, y

def sensored_sine(dur=10, dx=0.1):
	x = np.arange(0, dur, dx)
	y = np.sin(x)
	midpoint = len(x)/2
	y[midpoint-1/dx:midpoint+1/dx] = np.nan
	return x, y

def piecewise_constant(dur=100, dx=0.1, piece_len=30):
	x = np.arange(0, dur, dx)
	sig = np.sign(np.sin(x/(piece_len/np.pi)))
	sig += 1
	sig *= 5.0/2.0
	return x, sig


if __name__ == '__main__':
	test_spline()
