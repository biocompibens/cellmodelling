import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from multiprocessing import Pool
from scipy.spatial.distance import cdist
from .utils import cart2pol
from .utils import convert_angle
from .utils import pol2cart
from PIL import Image

from .warping import compute_warp, _create_sorted_contour
#import cv2


from scipy.ndimage import morphological_gradient




def _warp(args):
	sellabel, rot = args
	yxmaskD = np.argwhere(_cellsD == sellabel)
	yxborderD = np.argwhere(_cellsborderD == sellabel)
	yxmaskS = np.argwhere(_cellsS == sellabel)
	yxborderS = np.argwhere(_cellsborderS == sellabel)
	values = compute_warp(_imS, yxmaskS, yxmaskD, yxborderS, yxborderD, rotangle=rot)
	if not (values is None):
		return yxmaskD, values
	else:
		return None


def _wrapingPix(args):
	sellabel, rot,xmaskS,ymaskS = args
	if np.isnan(xmaskS):
		return np.nan,np.nan
	yxmaskD = np.argwhere(_cellsD == sellabel)
	xmin = np.min(yxmaskD[:,0])
	xmax = np.max(yxmaskD[:,0])
	ymin = np.min(yxmaskD[:,1])
	ymax = np.max(yxmaskD[:,1])

	yxborderD = np.argwhere(_cellsborderD == sellabel)
	yxmaskS = np.argwhere(_cellsS == sellabel)

	yxborderS = np.argwhere(_cellsborderS == sellabel)
	
	xyc = compute_warp_Pix(_imS,np.array([xmaskS,ymaskS]),yxborderS, yxborderD, rotangle=rot)

	return xyc

def readPixelPosition(filename):
	extension = filename[filename.rfind('.'):]	
	if 	extension == '.npy':
		centriolePos = np.array(np.load(filename))
	elif extension == '.csv':
		f = open(filename,'r')
		data = f.readlines()
		f.close()
		for iLine in xrange(1,len(data)):
			data[iLine] = data[iLine].split(',')
			data[iLine] = [int(data[iLine][0]),float(data[iLine][1]),float(data[iLine][2])]
		centriolePos = np.array(data)
	elif extension == '.tsv':
		f = open(filename,'r')
		data = f.readlines()
		f.close()
		for iLine in xrange(1,len(data)):
			data[iLine] = data[iLine].split('\t')
			data[iLine] = [int(data[iLine][0]),float(data[iLine][1]),float(data[iLine][2])]
		centriolePos = np.array(data)
	return centriolePos
		

def points_synthesis(fileName, labD, C, imS, labS, rotangle, rotangle_real, l_real, id, dir, n_jobs=1, display=False):
	global _cellsD, _cellsborderD, _cellsS, _cellsborderS ,_imS, _imD

	height, width, depth = imS.shape

	cellsS = labS # already reshaped
	cellsD = labD.reshape((height,width)).astype('int64')
	
	allborderS = (morphological_gradient(cellsS, size=3) > 0)
	cellsinsideS = cellsS.copy() 
	cellsinsideS[allborderS] = -1
	cellsinsideS[0, :] = -1
	cellsinsideS[:, 0] = -1
	cellsinsideS[cellsinsideS.shape[0]-1, :] = -1
	cellsinsideS[:, cellsinsideS.shape[1]-1] = -1
	cellsborderS = cellsS - cellsinsideS - 1	# -1 to compensate the previous -1s, this is to allow for one of the labels to be 0


	allborderD = (morphological_gradient(cellsD, size=3) > 0)
	cellsinsideD = cellsD.copy() 
	cellsinsideD[allborderD] = -1
	cellsinsideD[0, :] = -1
	cellsinsideD[:, 0] = -1
	cellsinsideD[cellsinsideD.shape[0]-1, :] = -1
	cellsinsideD[:, cellsinsideD.shape[1]-1] = -1
	cellsborderD = cellsD - cellsinsideD - 1
	
	alllabelsD = np.unique(labD)
	
	cellsborderDwhite = cellsborderD.astype('uint8')
	cellsborderDwhite[cellsborderDwhite<255]=0
	cellsborderDwhite=255-cellsborderDwhite

	
	#multiprocessing
	_cellsD = cellsD
	_cellsborderD = cellsborderD
	_cellsS = cellsS
	_cellsborderS = cellsborderS
	_imS = imS

	centriolePos = readPixelPosition(fileName)#np.array(np.load(fileName))
	_centriolPosition= np.empty((alllabelsD.shape[0],2))
	_centriolPosition.fill(np.nan)
	_centriolPosition[np.array([np.argwhere(id == i)[0][0] for i in centriolePos[:,0]]),:]=  centriolePos[:,1:3]

	parameters = [(int(alllabelsD[i]), float(rotangle_real[i]-rotangle[i])) for i in range(0, alllabelsD.shape[0], 1)]
	parametersCentriole = list(np.concatenate((parameters,np.copy(_centriolPosition)),axis=1))

	if n_jobs > 1:
		pool = Pool(n_jobs)
		xyc = list(pool.imap(_wrapingPix, parametersCentriole))
		pool.close()
		pool.join()
	else:
		xyc= list(map(_wrapingPix,parametersCentriole))

	_cellsD = None
	_cellsborderD = None
	_cellsS = None
	_cellsborderS = None
	_imS = None
	
	if display:
		imD = np.dstack((cellsborderDwhite,cellsborderDwhite,cellsborderDwhite))
		results = list(map(_warp, parameters))
		for i in range(0, alllabelsD.shape[0], 1):
			if not(results[i] is None):
				yxmaskD, values = results[i] 
				imD[yxmaskD[:,0], yxmaskD[:,1], :] = values
		for i in range(0,alllabelsD.shape[0],1):
			sellabel = alllabelsD[i]
			yxmaskD = np.argwhere(cellsD == sellabel)
			try:
				shift = 4
				center = tuple((C[i][::-1]/2**(-shift)).astype(int))
				axes = tuple((np.sqrt(l_real[i])[::-1]/2**(-shift)).astype(int)) # half size
				angle = int((180*rotangle[i]/np.pi))
				p0 = tuple(((C[i][::-1]+pol2cart(rotangle[i], np.sqrt(l_real[i][1])))/2**(-shift)).astype(int))
				cv2.ellipse(imD, center, axes, angle, 0, 360, (100,0,0, 150), thickness=1, lineType=cv2.CV_AA, shift=shift)

				cv2.circle(imD, tuple((C[i][::-1]/2**(-shift)).astype(int)), int(1/2**(-shift)), (200,0,0), thickness=2, lineType=cv2.CV_AA, shift=shift)
				if not(xyc[i] is None) and not np.isnan(xyc[i][0]):
					cv2.circle(imD, (np.floor(xyc[i][1]).astype(int),np.floor(xyc[i][0]).astype(int)), 3, (230, 138, 0), thickness=1, lineType=8, shift=0) 
			except np.linalg.LinAlgError as e:
				print('np.linalg.LinAlgError (image_synthesis): %s' % e)
		Image.fromarray(imD, 'RGB').save(dir+'fullwarping.tif')

	return imD,xyc

		
		
		

def compute_warp_Pix(imS, yxmaskS, yxborderS, yxborderD, rotangle=0):

	try:
		if (yxmaskS.shape[0] < 2) : 
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
			v1 = np.roll(yxAnchorS, -1, axis=0)
			v2 = yxAnchorS
			uu = v1[:, np.newaxis] - yxmaskS
			vv = v2[:, np.newaxis] - yxmaskS
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
			if pidx.shape[0] != 0:
				dist = cdist(yxmaskS[pidx], yxAnchorS)
				aidx = np.argsort(dist, axis=1)
				zidx = np.where(dist[range(aidx.shape[0]), aidx[:,0]] == 0)[0]
				nidx = np.where(dist[range(aidx.shape[0]), aidx[:,0]] > 0)[0]
			
				# weights set to 1 for points matching the anchors (distance 0)
				W[pidx[zidx]] = 0
				W[pidx[zidx], aidx[zidx,0]] = 1

				# weights set to linear interpolation for points between the anchors
				W[pidx[nidx]] = 0
				W[pidx[nidx,np.newaxis], aidx[nidx,:2]] = dist[np.arange(aidx.shape[0])[nidx,np.newaxis], aidx[nidx,:2]][:,[1,0]]

			# normalize weights
			W = np.divide(W,np.sum(W, axis=1)[:,np.newaxis])
			
			yx = np.dot(W, yxAnchorD)
			if yx.shape[0]>1:
				print( 'warning, plus d\'un centriole !? ')
			y = yx[:,1]
			x = yx[:,0]

			y[y >= imS.shape[0]-1] = imS.shape[0] - 1
			x[x >= imS.shape[1]-1] = imS.shape[1] - 1
	
			return np.concatenate((x,y))
	except ValueError as e: 
		print( 'warping.yp - compute_warp() - ValueError: %s' % e)
		#import pdb
		#pdb.set_trace()
		return None
	except IndexError as e: 
		print( 'warping.yp - compute_warp() - ValueError: %s' % e)
		#_create_sorted_contour(yxborderD[::-1], center=meanD, rot=-rotangle, N=100, display=True)
		#import pdb
		#pdb.set_trace()
		return None

