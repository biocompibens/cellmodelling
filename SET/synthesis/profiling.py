import matplotlib.pyplot as plt
import itertools
import numpy as np
import os
#import cv2
import pandas as pd
import sys


from scipy.spatial.distance import cdist
from skimage.measure import label
from scipy.ndimage import morphological_gradient
from multiprocessing import Pool
from PIL import Image, ImageDraw
from tqdm import tqdm

from .warping import compute_warp
from .utils import pol2cart, fit_all


_cellsS = None
_cellsborderS = None
_yxmaskD = None
_yxborderD = None
_imS = None


np.seterr(all='raise')

def _compute_singlecell(cellParam): 
	(_p,label, warp) = cellParam
	yxmaskS = np.argwhere(_cellsS == label) 
	yxborderS = np.argwhere(_cellsborderS == label)
	
	volumei = yxmaskS.shape[0] 
	borderi = (np.any(yxmaskS[:,0] == 0) or np.any(yxmaskS[:,1] == 0) or np.any(yxmaskS[:,0] == _cellsS.shape[0]-1) or np.any(yxmaskS[:,1] == _cellsS.shape[1]-1))
	if _p == '' or _p == 5 :
		Ci = np.mean(yxmaskS, axis=0)
		yxmaskS_centered = yxmaskS - Ci

		Si = np.dot(yxmaskS_centered.T, yxmaskS_centered) / yxmaskS_centered.shape[0] #covariance
	
		try:
			lr, vr = np.linalg.eigh(Si)
			eigenvalsi = lr
			rotanglei = np.arctan2(vr[0,1], vr[1,1])
		except np.linalg.LinAlgError as e:
			rotanglei = 0
			eigenvalsi = list([1,1])
			print('np.linalg.LinAlgError (profiling.py - _compute_singlecell): %s' % e)
		u1, u2, alpha, l1, l2, a1, a2, p = Ci[0],Ci[1],rotanglei,eigenvalsi[0],eigenvalsi[1],0,0,2
	else :
		u1, u2, alpha, l1, l2, a1, a2, p = fit_all(yxborderS, label, _p)
	
	if warp:
		values = compute_warp(_imS, yxmaskS, _yxmaskD, yxborderS, _yxborderD, rotangle=0)
	else:
		values = None
	
	return u1, u2, alpha, l1, l2, a1, a2, p, borderi, volumei, values
	
	
	


class Profiles(object):

	def __init__(self, df):
		self.df = df
		if self.df.columns.values.shape[0] > 10:
			rgb1dyx = self.df.columns.values[13:]
			self.yxmask = np.array([p.split(',')[1:] for p in rgb1dyx[:rgb1dyx.shape[0]//3]], dtype=int)
			self.mheight, self.mwidth = np.max(self.yxmask, axis=0) + 1 
		


	@classmethod 
	def from_image(cls, im_real, im_labels, radii=None, include_border=True, n_jobs=1, resolution=1, _p =''):
		global _yxmaskD, _yxborderD, _cellsS, _cellsborderS ,_imS

		process_warp = (radii != None)
		
		# add sanity checks : 1) image size; 2) size is a tuple of size 2; 3) im_real could be an option 
		
		if ((len(im_real.shape) != 3) or (im_real.shape[2] != 3)):
			raise ValueError('Profiles supports only RGB images')
		
		height, width = im_real.shape[:2]

		im_relabels = label(im_labels, background=0) # re-label from 0
		im_relabels = im_relabels-1 # -1 is background
		#im_relabels[im_relabels==-1] = -2 # -2 is background
	
		lid1 = np.dstack((im_relabels,im_labels))
		lid2 = lid1.reshape((im_labels.shape[0]*im_labels.shape[1],2))
		lid3 = np.unique(lid2, axis=0)
		lid4 = lid3[lid3[:,0]>=0,:] # remove borders
		lid = lid4[lid4[:,0].argsort(), 1]

		alllabels = np.sort(np.unique(im_relabels))
		alllabels = alllabels[alllabels>=0] # remove borders
		
		if process_warp:
			axes = 1+2*np.array(radii) # force axes to be odd
			mask = np.zeros(axes, dtype='uint8')
			im_mask = Image.fromarray(mask)
			draw = ImageDraw.Draw(im_mask)
			draw.ellipse((0,0,axes[1]-1,axes[0]-1), fill=1)
			mask = np.array(im_mask)
			yxmaskD = np.argwhere(mask > 0)	
			maskborder = (morphological_gradient(mask, size=3, mode='constant') > 0) & mask

		cells = im_relabels.copy() 
		cells[(morphological_gradient(im_relabels, size=3) > 0)] = -1 # cell contours set to -1
		cells[0, :] = -1
		cells[:, 0] = -1
		cells[cells.shape[0]-1, :] = -1
		cells[:, cells.shape[1]-1] = -1
		cellsborder = im_relabels - cells - 1	# -1 to compensate the previous -1s, this is to allow for one of the labels to be 0

		#multiprocessing		
		_cellsS = im_relabels
		_cellsborderS = cellsborder
		if process_warp:
			_yxmaskD = yxmaskD
			_yxborderD = np.argwhere(maskborder > 0)
			_imS = im_real
		
		parameters = [(_p,i, process_warp) for i in alllabels]
		if n_jobs > 1:

			pool = Pool(n_jobs)
			#results = list(tqdm(pool.imap_unordered(_compute_singlecell, parameters), total=len(parameters))) #chunksize=len(parameters)/n_jobs)))
			results = list(tqdm(pool.imap(_compute_singlecell, parameters), total=len(parameters))) #chunksize=len(parameters)/n_jobs)))
			pool.close()
			pool.join()
		else:
			results = list(tqdm(map(_compute_singlecell, parameters), total=len(parameters)))

		_cellsS = None
		_cellsborderS = None
		if process_warp:
			_yxmaskD = None
			_yxborderD = None
			_imS = None

		C = np.empty((alllabels.shape[0],2), dtype=float)
		L = np.empty((alllabels.shape[0],2), dtype=float)
		A = np.empty((alllabels.shape[0],2), dtype=float)
		q = np.empty(alllabels.shape[0], dtype=float)
		border = np.ones(alllabels.shape[0], dtype=bool)
		rotangle = np.empty(alllabels.shape[0], dtype=float)
		if process_warp:
			rgbmean = np.empty((alllabels.shape[0],3), dtype=float)
			rgb1Dprofile = np.empty((alllabels.shape[0],np.prod(yxmaskD.shape[0])*3), dtype=float)

		for i in alllabels:
			u1, u2, alpha, l1, l2, a1, a2, p, borderi, volumei, values  = results[i]
			C[i] = (u1, u2)
			L[i] = (l1*resolution**2, l2*resolution**2) 
			A[i] = (a1, a2)
			q[i] = p
			rotangle[i] = alpha
			border[i] = borderi
			if (process_warp and not(values is None)):
				rgbmean[i] = np.mean(values, axis=0)
				rgb1Dprofile[i] = np.reshape(values.T, np.prod(values.shape))

		if not include_border:
			notborder = ~border
			lid = lid[notborder]
			C = C[notborder]
			L = L[notborder]
			rotangle = rotangle[notborder]
			border = border[notborder]
			if process_warp:
				rgbmean = rgbmean[notborder]
				rgb1Dprofile = rgb1Dprofile[notborder]
				
		if process_warp:
			columns = ['id', 'x', 'y', 'theta', 'l1', 'l2', 'a1', 'a2', 'p', 'border', 'meanr', 'meang', 'meanb' ] + ['%s,%d,%d' % (('R','G','B')[i], y, x)  for i in range(3) for (y,x) in yxmaskD]
			data = np.hstack((lid[:,np.newaxis], C, rotangle[:,np.newaxis], L, A, q[:,np.newaxis], border[:,np.newaxis], rgbmean, rgb1Dprofile))	
		else:
			columns = ['id', 'x', 'y', 'theta', 'l1', 'l2', 'a1', 'a2', 'p', 'border']
			data = np.hstack((lid[:,np.newaxis], C, rotangle[:,np.newaxis], L, A, q[:,np.newaxis], border[:,np.newaxis]))	
		
		df = pd.DataFrame(index=np.arange(data.shape[0]), data=data, columns=columns)
		return cls(df)

	
	
	
	
	
	
	
