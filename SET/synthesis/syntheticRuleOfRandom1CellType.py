
#import line_profiler
import sys
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool
from scipy.spatial.distance import cdist, mahalanobis, euclidean

from scipy.ndimage.morphology import morphological_gradient 
from PIL import Image

from .drawing import image_synthesis
from .utils import fit_loc_angle, paramsAsMatrices, dG

from IPython import embed

_labels = None
_idxnn = None
_I = None
_border = None
_cellsborder = None
_params = None
_params_orig = None

def _min_dist(labi):
	try:
		#print labi
		idx = np.where(_labels==labi)[0]
		idist = []
		for j in _idxnn[labi]:
			#if _q==2:
			params = _params[j]
			params_orig = _params_orig[j]
			u, TI, A, q = paramsAsMatrices(np.concatenate((params, params_orig[3:])))
			#u, TI, A, q = paramsAsMatrices(params_orig)
			#idist.append(cdist(_I[idx],u.A,metric='mahalanobis', VI=VI))
			#idist.append(cdist(_I[idx], u.A, metric=dG, TI=TI, A=A, q=q)) 
			dd = []
			for v in _I[idx]:
				d = euclidean(u.A, v) 
				if d > 3*np.sqrt(np.linalg.det(TI.I)):
					dd.append(d)
				else:
					dd.append(dG(u.A, v, TI, A, q))
			
			idist.append(dd)
			#idist.append(cdist(u.A, _I[idx], metric=dG, TI=TI, A=A, q=q)) 
		
		#import pdb
		#pdb.set_trace()
			
		#idist = np.vstack(idist) # np.vstack(idist) and  axis=1 and exchange a and b in cdist
	except TypeError as e:
		print('%s: %s %s' % (e, labi, _labels))
	#return idx, _idxnn[labi][np.argmin(idist,axis=1)]
	return idx, _idxnn[labi][np.argmin(idist,axis=0)], np.min(idist,axis=0) # THIS IS THE GOOD ONE !!!

def _mean(labi):
	idxb = np.where(_cellsborder==labi)[0]
	if idxb.shape[0] <= 1 or _border[labi]:
		return _params[labi]
	else:
		
		params = _params[labi]
		params_orig = _params_orig[labi]
		
		u1, u2, alpha = fit_loc_angle(_I[idxb], params, params_orig, labi)
		'''
		alpha_prev = params[2]
		# force angle to be continuous
		if(np.abs(alpha_prev-alpha) > np.pi/2):
			if(alpha<alpha_prev):
				alpha += np.pi
			else:
				alpha -= np.pi
		if alpha > np.pi:
			alpha -= 2*np.pi
		if alpha < -np.pi:
			alpha += 2*np.pi
		'''
		#u1, u2 = np.mean(_I[idxb], axis=0)

		return u1, u2, alpha

#Lloyd_subpopRandom(I, C, S, labels, rotangle, border, l_real, rotangle_real, im_labels, im_real,cellSize, id, dir=None, prefix='000', n_jobs=1, max_iter=80): 

#@profile # doesnt play well with multiprocessing
def Lloyd_subpopRandom(I, labels, params, params_orig, border, im_labels_orig, im_values_orig, dir=None, prefix='000', n_jobs=1, max_iter=80): 
	global _labels, _idxnn, _I, _border, _cellsborder, _params, _params_orig
	
	_I = I
	_border = border
	_params_orig = params_orig
	

	width = im_values_orig.shape[1]
	height = im_values_orig.shape[0]
	NN = 20
	#embed()
	global boolCelluleSouche
	cellClass = np.load('/data/biocomp/laruelle/Xenopus_Walentek_st33/3/20x-st33-ctrl-3Class.npy')
	gobletSimuLabels= np.squeeze(np.unique(labels)[np.argwhere(cellClass!='Goblet')]) #cellClass!='SSC'
	boolCelluleSouche = [  ilabel in gobletSimuLabels for ilabel in np.unique(labels)]#[  not(ilabel in gobletSimuLabels) for ilabel in np.unique(labels)]

	l=0

	if dir : 
		im = image_synthesis(labels.reshape((height, width)), params, params_orig, im_labels_orig, im_values_orig, n_jobs=n_jobs, display=True)
		Image.fromarray(im, 'RGB').save('%s/shapes_%s_%04d.png' % (dir, prefix, l))
		Image.fromarray(labels.astype(np.uint32).reshape((height,width))).save('%s/labels_%s_%04d.png' % (dir, prefix, l))
	l+=1

	#return None, None
	
	moving=True
	maxmoving=0
	while (moving and maxmoving<max_iter):
		C = params[:,(0,1)]

		if maxmoving==1:
			#embed()

			'''
			#P30 : find stem cell with their size
			boolCelluleSouche=np.zeros(len(C))
			for eachCell in range(len(C)):
				yx = np.argwhere(labels==eachCell)
				#print len(yx)
				if len(yx)< 10000:# 
					boolCelluleSouche[eachCell]=1
			boolCelluleSouche = np.array(boolCelluleSouche)
			'''
			#embed()
			#move closest stem cell away
			if 1 : #for random position of sub cell population
				for eachC in range(len(C)): #range(1,len(C)) --> pourquoi 1 ?? je le faisais pas pour ependyme 
					if boolCelluleSouche[eachC]==1 and not(border[eachC]) :
						C[eachC,:] = [np.random.randint(width+1),np.random.randint(height+1)]
						#print eachC
			if 0 : #for shuffle position of sub cell population
				toShuffle = []
				#embed()
				for eachC in range(len(C)):
					if boolCelluleSouche[eachC]==1 and not(border[eachC]) :
						toShuffle.append(eachC)
				toShuffleInitial = np.copy(toShuffle)
				shuffle(toShuffle)
				C[toShuffleInitial,:] =C[toShuffle,:]


			labels = np.argmin(cdist(I, C), axis=1) #voronoi

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
		_labels = None
		_idxnn = None
		
		# newdistmap=np.zeros(labels.shape, dtype=float) # temp
		newlabels=np.zeros(labels.shape, dtype=int)
		for i in range(C.shape[0]):
			idx, lab, dist = results[i]
			
			if idx.shape[0] == 0:
				newlabels[int(C[i,0]*width+C[i,1])] = i
				print('label',i, ' has 1px')
			else :#add by elise
				newlabels[idx] = np.squeeze(lab)
			# newdistmap[idx] = dist # temp
			
		labels = newlabels
		
		# if dir != None:
			# Image.fromarray((newdistmap*1000).astype(np.uint16).reshape((height,width))).save('%s/%s_%04d_dist.tif' % (dir, prefix, l))
		#break
		
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
			pool = Pool(n_jobs)
			results = list(pool.imap(_mean, parameters))
			pool.close()
			pool.join()
		else:
			results = list(map(_mean, parameters))
		_cellsborder = None
		
		params=np.array(results)
		newC = params[:,(0,1)]
		
		moving = np.linalg.norm(C-newC, ord=np.inf) > 0
		maxmoving+=1
		
		if dir != None:
			#im = image_synthesis(im_labels, params, params_orig, im_labels_orig, im_values_orig, n_jobs=n_jobs, display=True)
			#Image.fromarray(im, 'RGB').save('%s/%s_%04d.png' % (dir, prefix, l))
			Image.fromarray(labels.astype(np.uint32).reshape((height,width))).save('%s/%s_%04d.png' % (dir, prefix, l))
		
		l+=1
		
	if dir != None:
		#im = image_synthesis(im_labels, params, params_orig, im_labels_orig, im_values_orig, n_jobs=n_jobs, display=False)
		#Image.fromarray(im, 'RGB').save('%s/%s_%04d.png' % (dir, prefix, l))
		Image.fromarray(labels.astype(np.uint32).reshape((height,width))).save('%s/%s_%04d.png' % (dir, prefix, l))
	
	
	# u1, u2, alpha, l1, l2, a1, a2, p = params_orig[88]
	# u, TI, A, p = paramsAsMatrices(params_orig[88])
	# print('l1:',l1)
	# print('l2:',l2)
	# print('l1*l2:', l1*l2)
	# print('det(T):', np.linalg.det(TI.I))
	
	#dist88 = cdist(u.A, I, metric=dG, TI=TI, A=A, q=p) * 100
	#Image.fromarray(dist88.astype(np.uint16).reshape((height,width))).save('%s/dist88_%s_%04d.tif' % (dir, prefix, l))
	
	_I = None
	_border = None
	_params_orig = None

	return params, labels



	
