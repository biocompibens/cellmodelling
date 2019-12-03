

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from .utils import cart2pol
from .utils import convert_angle
from .utils import _create_sorted_contour

#np.seterr(all='raise') # to raise excpeption even for warnings

def warp(imS, imD, yxmaskS, yxmaskD, yxborderS, yxborderD, rotangle=0):
	result = compute_warp(imS, yxmaskS, yxmaskD, yxborderS, yxborderD, rotangle=0)
	if not(result is None): 
		#redraw
		imD[yxmaskD[:,0], yxmaskD[:,1], :] = result
	
def compute_warp(imS, yxmaskS, yxmaskD, yxborderS, yxborderD, rotangle=0):

	try:
		if ((yxmaskS.shape[0] < 2) or (yxmaskD.shape[0] < 2)): 
			return None

		meanS = np.mean(yxborderS, axis=0)	
		meanD = np.mean(yxborderD, axis=0)	
		
		yxAnchorS = _create_sorted_contour(yxborderS[::-1], center=meanS, rot=0,         N=100)
		yxAnchorD = _create_sorted_contour(yxborderD[::-1], center=meanD, rot=-rotangle, N=100)
				
		if (yxAnchorS is None) or (yxAnchorD is None):
			return None
		
		if (np.unique(yxAnchorS, axis=0).shape[0] < 3) or (np.unique(yxAnchorD, axis=0).shape[0] < 3):
			return None
		
		with np.errstate(invalid='ignore'): # prevent from printing division by zero warnings (these are managed by special cases below)
		
			# computing weights
			v1 = np.roll(yxAnchorD, -1, axis=0)
			v2 = yxAnchorD
			uu = v1[:, np.newaxis] - yxmaskD
			vv = v2[:, np.newaxis] - yxmaskD
			nu = np.linalg.norm(uu, axis=2).T
			nv = np.linalg.norm(vv, axis=2).T
			nuv = nu*nv
			sin_uv = np.cross(uu,vv,axis=2).T / nuv
			cos_uv = np.einsum('ijk,ijk->ji',uu,vv) / nuv
			_tan1 = sin_uv / (1+cos_uv)
			_tan2 = np.roll(_tan1, 1, axis=1)
			W = (_tan1 + _tan2) / nv 
		
			# identify special cases 
			pidx = np.unique(np.where(((sin_uv == 0) & (cos_uv < 0)) | np.isnan(W))[0]) 
			dist = cdist(yxmaskD[pidx], yxAnchorD)
			aidx = np.argsort(dist, axis=1)
			zidx = np.where(dist[list(range(aidx.shape[0])), aidx[:,0]] == 0)[0]
			nidx = np.where(dist[list(range(aidx.shape[0])), aidx[:,0]] > 0)[0]
			
			# weights set to 1 for points matching the anchors (distance 0)
			W[pidx[zidx]] = 0
			W[pidx[zidx], aidx[zidx,0]] = 1

			# weights set to linear interpolation for points between the anchors
			W[pidx[nidx]] = 0
			W[pidx[nidx,np.newaxis], aidx[nidx,:2]] = dist[np.arange(aidx.shape[0])[nidx,np.newaxis], aidx[nidx,:2]][:,[1,0]]

			# normalize weights
			W = np.divide(W,np.sum(W, axis=1)[:,np.newaxis])
			
			yx = np.dot(W, yxAnchorS)

			y = yx[:,0] 
			x = yx[:,1] 

			yf = np.floor(y).astype(int)
			xf = np.floor(x).astype(int)
			yf[yf >= imS.shape[0]-1] = imS.shape[0] - 1
			xf[xf >= imS.shape[1]-1] = imS.shape[1] - 1
			
			xf_1 = xf.copy()+1
			xf_1[xf_1 >= imS.shape[1]-1] = imS.shape[1] - 1
			yf_1 = yf.copy()+1
			yf_1[yf_1 >= imS.shape[0]-1] = imS.shape[0] - 1
			
			wy = y-yf
			wx = x-xf
			w00 = (1-wy)*(1-wx)
			w10 = wy*(1-wx)
			w01 = (1-wy)*wx
			w11 = wy*wx
	
			return np.transpose(np.multiply(w00,imS[yf, xf, :].T) + np.multiply(w10,imS[yf_1, xf, :].T) + np.multiply(w01,imS[yf, xf_1, :].T) + np.multiply(w11,imS[yf_1, xf_1, :].T))
	except ValueError as e: 
		print('warping.yp - compute_warp() - ValueError: %s' % e)
		#import pdb
		#pdb.set_trace()
		return None
	except IndexError as e: 
		print('warping.yp - compute_warp() - ValueError: %s' % e)
		#_create_sorted_contour(yxborderD[::-1], center=meanD, rot=-rotangle, N=100, display=True)
		#import pdb
		#pdb.set_trace()
		return None

		

