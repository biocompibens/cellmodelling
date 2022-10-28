
import numpy as np
#import cv2

from multiprocessing import Pool
from scipy.ndimage import morphological_gradient
from .warping import compute_warp
from .utils import pol2cart, paramsAsMatrices, dG
from scipy.optimize import minimize
import scipy.ndimage

_cellsD = None
_cellsborderD = None
_cellsS = None
_cellsborderS = None
_imS = None

def level_to_poly(u, TI, A, p, level=1, N=50):
	def err(rr,tt,TI,A,p,level):
		x,y = pol2cart(tt, rr[0])
		return np.power(dG(np.array([0,0]), np.array([x,y]), TI=TI, A=A, q=p) - level, 2)[0]
	theta = np.arange(-np.pi,np.pi, 2*np.pi/N)
	res = []
	for t in theta:
		min_res = minimize(err, x0=1, bounds=((0, None),),	args=(t,TI,A,p,level))
		# if not min_res.success:
			# import pdb
			# pdb.set_trace()
		res.append(min_res.x[0])
	return u + np.array(pol2cart(theta, np.array(res))).T  # removed transpose u


def _warp(xxx_todo_changeme):
	(sellabel, rot) = xxx_todo_changeme
	yxmaskD = np.argwhere(_cellsD == sellabel)
	yxborderD = np.argwhere(_cellsborderD == sellabel)
	yxmaskS = np.argwhere(_cellsS == sellabel)
	yxborderS = np.argwhere(_cellsborderS == sellabel)
	values = compute_warp(_imS, yxmaskS, yxmaskD, yxborderS, yxborderD, rotangle=rot)
	if not (values is None):
		return yxmaskD, values
	else:
		return None


def label2Contours(ImgL1):
	#region_borders = imdilate(lblImg,ones(3,3)) > imerode(lblImg,ones(3,3)); #https://stackoverflow.com/questions/5265837/find-outlines-borders-of-label-image-in-matlab
	#https://stackoverflow.com/questions/34771798/isolating-coastal-grids-using-convolving-window-python
	#scipy.ndimage.morphology.binary_dilation(ImgL1,structure = np.ones((3,3)))>
	region_borders = scipy.ndimage.morphological_gradient(ImgL1,
                                            footprint=[[0, 1, 0],
                                                       [1, 1, 1],
                                                       [0, 1, 0]])
	yxB = np.argwhere(region_borders!=0)
	region_borders[yxB[:,0],yxB[:,1]]=255
	#Image.fromarray(region_borders.astype(np.uint32)).save('testMorphoGradient.png')
	return region_borders 

def image_synthesis(im_labels, params, params_orig, im_labels_orig, im_values_orig, n_jobs=1,pShape = '', display=False):
	global _cellsD, _cellsborderD, _cellsS, _cellsborderS ,_imS, _imD

	height, width, depth = im_values_orig.shape

	cellsS = im_labels_orig 
	cellsD = im_labels
	
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
	
	#alllabelsS = np.unique(labS)
	alllabelsD = np.unique(im_labels)
	
	# import pdb
	# pdb.set_trace()
	
	'''
	#imD = np.zeros((height, width, 3), dtype='uint8')
	cellsborderDwhite = cellsborderD.astype('uint8')
	cellsborderDwhite[cellsborderDwhite<255]=0
	cellsborderDwhite=255-cellsborderDwhite
	imD = np.dstack((cellsborderDwhite,cellsborderDwhite,cellsborderDwhite))
	'''
	rotangle = params[:,2] # temp
	rotangle_real = params_orig[:,2] # temp
	#	if not display:
	imD = np.zeros((height, width, 3), dtype='uint8')
	if display :
		# multiprocessing
		_cellsD = cellsD
		_cellsborderD = cellsborderD
		_cellsS = cellsS
		_cellsborderS = cellsborderS
		_imS = im_values_orig
	
	
		parameters = [(int(alllabelsD[i]), float(rotangle_real[i]-rotangle[i])) for i in range(0, alllabelsD.shape[0], 1)]
		if n_jobs > 1:
			pool = Pool(n_jobs)
			results = list(pool.imap(_warp, parameters))
			pool.close()
			pool.join()
		else:
			results = list(map(_warp, parameters))
		for i in range(0, alllabelsD.shape[0], 1):
			if not(results[i] is None):
				yxmaskD, values = results[i] 
				imD[yxmaskD[:,0], yxmaskD[:,1], :] = values

	_cellsD = None
	_cellsborderD = None
	_cellsS = None
	_cellsborderS = None
	_imS = None

	
	#else:
	
	# imD = np.zeros((height, width, 3), dtype='uint8')
	# cellsborderDwhite = cellsborderD.astype('uint8')
	# cellsborderDwhite[cellsborderDwhite<255]=0
	# cellsborderDwhite=255-cellsborderDwhite
	# imD[:,:,0] = cellsborderDwhite
	# imD[:,:,1] = cellsborderDwhite
	# imD[:,:,2] = cellsborderDwhite
	# imD = np.dstack((cellsborderDwhite, cellsborderDwhite, cellsborderDwhite))
	if pShape != '':
		from tqdm import tqdm
		imE = np.zeros((height, width, 3), dtype='uint8')
		imE[:,:,2] = label2Contours(im_labels.reshape(height, width))
		for i in tqdm(range(0,alllabelsD.shape[0],1)):
		
			sellabel = alllabelsD[i]
			yxmaskD = np.argwhere(cellsD == sellabel)

			u1, u2, alpha = params[i]
			u10, u20, alpha0, l10, l20, a10, a20, p0 = params_orig[i]
			#try:
			u, TI, A, p = paramsAsMatrices(np.concatenate((params[i], params_orig[i][3:]))) 
			#u, TI, A, p = paramsAsMatrices(params_orig[i]) # TO BE REMOVED : ORIG PARAM ONLY
			#except:
			#	import pdb
			#	pdb.set_trace()
		
			C = u.A[0] # temp
			l_real = np.array((l10, l20)) # temp

		
			try:
				shift = 4
				center = tuple((C[::-1]/2**(-shift)).astype(int))
				axes = tuple((l_real/2**(-shift)).astype(int)) 
				angle = int((180*rotangle[i]/np.pi))
				p1 = tuple(((C[::-1]+pol2cart(-rotangle[i], l_real[0]))/2**(-shift)).astype(int))
				#cv2.ellipse(imE, center, axes, angle, 0, 360, (0,128,128, 255), thickness=1, lineType=cv2.LINE_AA, shift=shift)

				poly = level_to_poly(C, TI, A, p, level=0.5, N=50) # BETTER EXPEND THIS!!
				poly = (poly[:,::-1]/2**(-shift)).astype(int)
				cv2.polylines(imE, [poly.astype(int)], True, (255,0,0,255), thickness=1, lineType=cv2.CV_AA, shift=shift)

				#cv2.line(imE, center, p1, (0,0,255, 255), thickness=1, lineType=cv2.LINE_AA, shift=shift)
				##cv2.putText(imE, '%s : %d' % (str(tuple((np.array(l_real)/(2*np.pi)).astype(int))), yxmaskD.shape[0]), tuple(C[i][::-1].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.25, color=(200,0,0), thickness = 1, lineType=cv2.LINE_AA)
				##cv2.putText(imE, '%.2f' % rotangle[i], tuple(C[::-1].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.25, color=(200,0,0), thickness = 1, lineType=cv2.LINE_AA)
				#radius = np.pi*np.sqrt(np.product(l_real))/2
				cv2.circle(imE, tuple((C[::-1]/2**(-shift)).astype(int)), int(1/2**(-shift)), (0,255, 0), thickness=2, lineType=cv2.CV_AA, shift=shift)
				#if i==iCCC:
				#	cv2.circle(imE, tuple((C[::-1]/2**(-shift)).astype(int)), int(1/2**(-shift)), (255,0,0), thickness=3, lineType=cv2.LINE_AA, shift=shift)
			except np.linalg.LinAlgError as e:
				print(('np.linalg.LinAlgError (image_synthesis): %s' % e))
	else : 
		imE = np.zeros((height, width, 3), dtype='uint8')
		imE[:,:,2] = label2Contours(im_labels.reshape(height, width))
		C = params[:,(0,1)]
		l_real = params_orig[:,(3,4)]
		for i in range(0,alllabelsD.shape[0],1):
			sellabel = alllabelsD[i]
			yxmaskD = np.argwhere(cellsD == sellabel)
			try:
				shift = 4
				center = tuple((C[i][::-1]/2**(-shift)).astype(int))
				axes = tuple((np.sqrt(l_real[i])[::-1]/2**(-shift)).astype(int)) # half size
				##axes = tuple((2*np.sqrt(l_real[i])[::-1]/2**(-shift)).astype(int)) # right size
				angle = int((180*rotangle[i]/np.pi))#int((180*np.arctan2(v[0,0], v[1,0])/np.pi))
				p0 = tuple(((C[i][::-1]+pol2cart(rotangle[i], np.sqrt(l_real[i][1])))/2**(-shift)).astype(int))
				cv2.ellipse(imE, center, axes, angle, 0, 360, (100,0,0, 150), thickness=1, lineType=cv2.CV_AA, shift=shift)
				##cv2.line(imE, center, p0, (200,200,200, 125), thickness=1, lineType=cv2.LINE_AA, shift=shift)
				##cv2.putText(imE, '%s : %d' % (str(tuple((np.array(l_real[i])/(2*np.pi)).astype(int))), yxmaskD.shape[0]), tuple(C[i][::-1].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.25, color=(200,0,0), thickness = 1, lineType=cv2.LINE_AA)
				##cv2.putText(imE, '%.2f' % rotangle[i], tuple(C[i][::-1].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.25, color=(200,0,0), thickness = 1, lineType=cv2.LINE_AA)
				radius = np.pi*np.sqrt(np.product(l_real[i]))/2
				##cv2.circle(imE, tuple((C[i][::-1]/2**(-shift)).astype(int)), int(radius/20/2**(-shift)), (255,0,0), thickness=1, lineType=cv2.LINE_AA, shift=shift)
				cv2.circle(imE, tuple((C[i][::-1]/2**(-shift)).astype(int)), int(1/2**(-shift)), (200,0,0), thickness=2, lineType=cv2.CV_AA, shift=shift)
				#if i==iCCC:
				#	cv2.circle(imE, tuple((C[i][::-1]/2**(-shift)).astype(int)), int(1/2**(-shift)), (255,0,0), thickness=3, lineType=cv2.LINE_AA, shift=shift)
			except np.linalg.LinAlgError as e:
				print('np.linalg.LinAlgError (image_synthesis): %s' % e)
	return imD,imE


		
		
		
