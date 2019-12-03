import sys
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool
from scipy.spatial.distance import cdist, mahalanobis, euclidean

from scipy.ndimage.morphology import morphological_gradient 
from PIL import Image

from .drawing import image_synthesis
from .utils import fit_loc_angle, paramsAsMatrices, dG

_labels = None
_idxnn = None
_I = None
_border = None
_cellsborder = None
_params = None
_params_orig = None


def _min_dist(labi):
	try:
		idx = np.where(_labels==labi)[0]
		idist = []
		for j in _idxnn[labi]:
			params = _params[j]
			params_orig = _params_orig[j]
			u, TI, A, q = paramsAsMatrices(np.concatenate((params, params_orig[3:])))

			dd = []
			for v in _I[idx] :
				d = euclidean(u.A, v) 
				if d > 3*np.sqrt(np.linalg.det(TI.I)):
					dd.append(d)
				else:
					dd.append(dG(u.A, v, TI, A, q))
		
			idist.append(dd)

	except TypeError as e:
		print('%s: %s %s' % (e, labi, _labels))
	return idx, _idxnn[labi][np.argmin(idist,axis=0)], np.min(idist,axis=0) # THIS IS THE GOOD ONE !!!

def _mean(labi):
	idxb = np.where(_cellsborder==labi)[0]
	if idxb.shape[0] <= 1 or _border[labi]:
		return _params[labi]
	else:
		
		params = _params[labi]
		params_orig = _params_orig[labi]
		
		u1, u2, alpha = fit_loc_angle(_I[idxb], params, params_orig, labi)

		return u1, u2, alpha

	

def Lloyd(I, labels, params, params_orig, border, im_labels_orig, im_values_orig, dir=None, prefix='000', n_jobs=1, max_iter=80,pShape = ''): 
	global _labels, _idxnn, _I, _border, _cellsborder, _params, _params_orig,_5p

	_I = I
	_border = border
	_labels = labels
	_params_orig = params_orig
	_5p = pShape

	
	doWarping = im_values_orig.sum()

	width = im_values_orig.shape[1]
	height = im_values_orig.shape[0]
	NN = 20
		
	l=0

	if dir : 
		Image.fromarray(labels.astype(np.uint32).reshape((height,width))).save('%s/labels_%s_%04d.png' % (dir, prefix, l))
		im,imE = image_synthesis(labels.reshape((height, width)), params, params_orig, im_labels_orig, im_values_orig, n_jobs=n_jobs, pShape=pShape,display=doWarping)
		Image.fromarray(imE, 'RGB').save('%s/shapes_%s_%04d.png' % (dir, prefix, l))
		if doWarping:
			Image.fromarray(im, 'RGB').save('%s/morph_%s_%04d.png' % (dir, prefix, l))
		

	l+=1

	moving=True
	maxmoving=0
	while (moving and maxmoving<max_iter):
		C = params[:,(0,1)]
		distC = cdist(C,C,metric='euclidean')
		idxnn = np.argsort(distC,axis=1)[:,:NN]
		
		# multiprocessing
		_params = params
		_labels = labels
		_idxnn = idxnn
		parameters = list(range(C.shape[0]))
		if n_jobs > 1:
			pool = Pool(n_jobs)
			results = list(pool.imap(_min_dist, parameters))
			pool.close()
			pool.join()
		else:
			results = list(map(_min_dist, parameters))	
		_idxnn = None
		
		newlabels=np.zeros(labels.shape, dtype=int)
		for i in range(C.shape[0]):
			idx, lab, dist = results[i]
			
			if idx.shape[0] == 0:
				newlabels[int(C[i,0]*width+C[i,1])] = i
				print('label',i, ' has 1px')
			else :
				newlabels[idx] = np.squeeze(lab)
			
		labels = newlabels

		
		im_labels = labels.reshape((height, width))
		allborder = (morphological_gradient(im_labels, size=3) > 0)
		cellsinside = im_labels.copy() 
		cellsinside[allborder] = -1
		cellsinside[0, :] = -1
		cellsinside[:, 0] = -1
		cellsinside[cellsinside.shape[0]-1, :] = -1
		cellsinside[:, cellsinside.shape[1]-1] = -1
		cellsborder = im_labels - cellsinside - 1
		
		_cellsborder = cellsborder.reshape(width*height)
		parameters = list(range(C.shape[0]))
		if n_jobs > 1:
			try : 
				pool = Pool(n_jobs)
				results = list(pool.imap(_mean, parameters))
				pool.close()
				pool.join()
			except :
				results = list(map(_mean, parameters))
		else:
			results = list(map(_mean, parameters))
		_cellsborder = None
		
		params=np.array(results)#[:,(0,1,2)]
		_labels = None

		newC = params[:,(0,1)]
		
		moving = np.linalg.norm(C-newC, ord=np.inf) > 0
		maxmoving+=1
		
		if dir != None:
			Image.fromarray(labels.astype(np.uint32).reshape((height,width))).save('%s/labels_%s_%04d.png' % (dir, prefix, l))
			im,imE = image_synthesis(labels.reshape((height, width)), params, params_orig, im_labels_orig, im_values_orig, n_jobs=n_jobs, pShape=pShape,display=doWarping)
			Image.fromarray(imE, 'RGB').save('%s/shapes_%s_%04d.png' % (dir, prefix, l))
			if doWarping:
				Image.fromarray(im, 'RGB').save('%s/morph_%s_%04d.png' % (dir, prefix, l))


		l+=1
		
	if dir != None:
		Image.fromarray(labels.astype(np.uint32).reshape((height,width))).save('%s/labels_%s_%04d.png' % (dir, prefix, l))
		im,imE = image_synthesis(labels.reshape((height, width)), params, params_orig, im_labels_orig, im_values_orig, n_jobs=n_jobs, pShape=pShape,display=doWarping)
		Image.fromarray(imE, 'RGB').save('%s/shapes_%s_%04d.png' % (dir, prefix, l))
		if doWarping:
			Image.fromarray(im, 'RGB').save('%s/morph_%s_%04d.png' % (dir, prefix, l))

	

	_I = None
	_border = None
	_params_orig = None

	return params, labels



	
