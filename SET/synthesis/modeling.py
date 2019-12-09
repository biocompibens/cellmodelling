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
_S_1 = None

def _min_dist_5p(labi):

	idx = np.where(_labels==labi)[0]
	idist = []

	for j in _idxnn[labi]: #take each of the 20 closest centers of the label = labi
		idist.append(cdist(_I[idx],_params[j,:2][np.newaxis,:],metric='mahalanobis', VI=_S_1[j])) #VI : ndarray The inverse of the covariance matrix for Mahalanobis
	idist = np.hstack(idist)

	return idx, _idxnn[labi][np.argmin(idist,axis=1)], np.min(idist,axis=0) 

def _mean_5p(labi):

	idx = np.where(_labels==labi)[0]
	if idx.shape[0] <= 1 or _border[labi]: # if small label or at borders
		return _params[labi,0].copy(),_params[labi,1].copy(), _params[labi,2].copy()
	else:
		Ci = np.mean(_I[idx], axis=0)
		try:
			
			yxcelli = _I[idx] - Ci
			Sj = np.dot(yxcelli.T, yxcelli) / yxcelli.shape[0]
			lj, vj = np.linalg.eigh(Sj)
			alphaj = np.arctan2(vj[0,1], vj[1,1])

			# force angle to be continuous
			if(np.abs(_params[labi,2]-alphaj) > np.pi/2):
				if(alphaj<_params[labi,2]):
					alphaj += np.pi
				else:
					alphaj -= np.pi
			if alphaj > np.pi:
				alphaj -= 2*np.pi
			if alphaj < -np.pi:
				alphaj += 2*np.pi

		except np.linalg.LinAlgError as e:
			print('np.linalg.LinAlgError (modeling.py - _mean): %s' % e)
			return Ci[0],Ci[1],_params[labi,2].copy()#, np.identity(2)
		return Ci[0],Ci[1], alphaj

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
	global _labels, _idxnn, _I, _border, _cellsborder, _params, _params_orig,_5p,_S_1

	_I = I
	_border = border
	_labels = labels
	_params_orig = params_orig
	_5p = pShape

	#reconstruct S from angles_orig and l_orig
	_P = [np.array([[np.cos(alpha),-np.sin(alpha)],[np.sin(alpha),np.cos(alpha)]]) for alpha in params_orig[:,2]] # set rotation matrices
	S_orig = np.array([np.dot(np.dot(np.linalg.inv(_p),np.identity(2)*_l),_p) for _l, _p in zip(params_orig[:,(3,4)], _P)]) # rotate and rescale
	S_1 = []
	for i in range(S_orig.shape[0]):
		try:
			S_1.append(np.linalg.inv(S_orig[i]))
		except np.linalg.LinAlgError as e:
			S_1.append(np.identity(2))
			print('np.linalg.LinAlgError (modeling.py - Lloyd) on cell %d: %s' % (i,e))
	_S_1 = np.array(S_1).copy()

	doWarping = im_values_orig.sum()

	width = im_values_orig.shape[1]
	height = im_values_orig.shape[0]
	NN = 20
		
	l=0

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
			if _5p == '':
				results = list(pool.imap(_min_dist_5p, parameters))#_5p
			else : 
				results = list(pool.imap(_min_dist, parameters))
			pool.close()
			pool.join()
		else:
			if _5p == '':
				results = list(map(_min_dist_5p, parameters))		
			else : 
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
		_labels = labels
		
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
				if _5p == '':
					results =  list(pool.imap(_mean_5p, parameters))#_5p
				else : 
					results = list(pool.imap(_mean, parameters))
				pool.close()
				pool.join()
			except :
				if _5p == '':
					results =  list(map(_mean_5p, parameters))#_5p
				else :
					results = list(map(_mean, parameters))
		else:
			if _5p == '':
				results =  list(map(_mean_5p, parameters))#_5p
			else :
				results = list(map(_mean, parameters))
		_cellsborder = None
		
		params=np.array(results)
		_labels = None

		rotMatrix = [np.array([[np.cos(alpha_l),-np.sin(alpha_l)],[np.sin(alpha_l),np.cos(alpha_l)]]) for alpha_l in params[:,2]]
		S_iter = np.array([np.dot(np.dot(np.linalg.inv(_p),np.identity(2)*_l),_p) for _l, _p in zip(params_orig[:,(3,4)], rotMatrix)])
		S_1 = []
		for i in range(S_iter.shape[0]):
			try:
				S_1.append(np.linalg.inv(S_iter[i]))
			except np.linalg.LinAlgError as e:
				S_1.append(np.identity(2))
				print('np.linalg.LinAlgError (modeling.py - Lloyd) on cell %d: %s' % (i,e))
		_S_1 = np.array(S_1).copy()
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
	_S_1 = None
	return params, labels



	
