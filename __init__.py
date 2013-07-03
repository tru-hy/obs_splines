"""
Package for some "obscure"/modern splines

Based on http://arxiv.org/abs/1208.2292 and
http://dx.doi.org/10.1016/j.csda.2009.09.020
"""

import numpy as np
from scipy.fftpack import dct, idct
from scipy.interpolate import interp1d

def get_mangled_diff_diagonal(n, s):
	# See eqn 11 in http://arxiv.org/abs/1208.2292
	rng = np.arange(n)
	n = float(n)
	diag = 1.0/(1 + s*(-2 + 2*np.cos(rng*np.pi/n))**2)
	return diag

def smooth1d_grid_l2_l2_simple(y, smoothing=1.0):
	"""Presented in http://dx.doi.org/10.1016/j.csda.2009.09.020"""
	diag = get_mangled_diff_diagonal(len(y), smoothing)
	return idct(diag*dct(y, norm='ortho'), norm='ortho')


def smooth1d_grid_l2_l2_weighted(y, weights=None, smoothing=1.0, crit=1e-3, max_iters=100):
	"""Presented in http://dx.doi.org/10.1016/j.csda.2009.09.020"""
	diag = get_mangled_diff_diagonal(len(y), smoothing)

	valid = np.isfinite(y)
	
	if weights is not None:
		w = valid*weights
	elif np.all(valid):
		# Without missing values/weights
		return idct(diag*dct(y, norm='ortho'), norm='ortho')
	else:
		w = valid

	valid_y = y.copy()
	rng = np.arange(len(valid_y))

	# We could maybe do better with the guess
	invalid = ~valid
	valid_y[invalid] = interp1d(rng[valid], valid_y[valid])(rng[invalid])
	z = valid_y
	prev_z = None
	for i in range(max_iters):
		z = idct(diag*dct(w*(valid_y - z) + z, norm='ortho'), norm='ortho')
		if prev_z is None:
			prev_z = z
			continue
		change, prev_z = z - prev_z, z
		if np.max(np.abs(change)) < crit:
			break

	
	return z
	
	
def shrink(v, invs):
	absvinvs = np.abs(v) - invs
	return np.sign(v)*(absvinvs)*(absvinvs > 0)

def smooth1d_grid_l1_l2(y, smoothing=1.0, crit=1e-3, max_iters=1000):
	"""
	Based on http://arxiv.org/abs/1208.2292
	Fixed N_i = 1
	"""
	n = len(y)
	d = np.zeros(n)
	b = np.zeros(n)
	prev_z = None
	
	diag = get_mangled_diff_diagonal(n, smoothing)
	n = float(n)
	for k in range(max_iters):
		# Equivalent, but avoid recalculation of the
		# diff diagonal eigenvaluething
		#z = smooth1d_grid_l2_l2(d + y - b, smoothing)
		z = idct(diag*dct(d + y - b, norm='ortho'), norm='ortho')

		d = shrink(z - y + b, 1.0/(smoothing))
		b += z - y - d
		
		if prev_z is None:
			prev_z = z
			continue
		
		change = np.max(np.abs((z - prev_z)))
		
		prev_z = z
		if change < crit:
			break
	
	return z

def smooth1d_grid_l1_l2_missing(y, smoothing=1.0, crit=1e-3, max_iters=1000):
	"""
	Based on http://arxiv.org/abs/1208.2292
	Fixed N_i = 1
	"""
	n = len(y)
	d = np.zeros(n)
	b = np.zeros(n)
	
	valid = np.isfinite(y)
	invalid = ~valid
	rng = np.arange(n)
	valid_y = y.copy()
	valid_y[invalid] = interp1d(rng[valid], valid_y[valid])(rng[invalid])
	
	z = valid_y
	prev_z = None
	
	diag = get_mangled_diff_diagonal(n, smoothing)
	n = float(n)
	for k in range(max_iters):
		# Equivalent, but avoid recalculation of the
		# diff diagonal eigenvaluething
		#z = smooth1d_grid_l2_l2(d + y - b, smoothing)
		z = idct(diag*dct(d + valid_y - b, norm='ortho'), norm='ortho')

		d[valid] = shrink(z[valid] - y[valid] + b[valid], 1.0/(smoothing))
		b[valid] += z[valid] - y[valid] - d[valid]
		
		if prev_z is None:
			prev_z = z
			continue
		
		change = np.max(np.abs((z - prev_z)))
		
		prev_z = z
		if change < crit:
			break
	
	return z
	
