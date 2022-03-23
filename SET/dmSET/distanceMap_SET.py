import os,sys
from multiprocessing import Pool

from PIL import Image
import itertools
from scipy.ndimage import morphological_gradient,distance_transform_edt, distance_transform_cdt
from scipy.signal import medfilt
from skimage.morphology import medial_axis,skeletonize
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist


import ctypes 
import time

class setImage(ctypes.Structure) :
	_fields_ = [("nbPxS",ctypes.POINTER(ctypes.c_int)),
				("nbPxC",ctypes.POINTER(ctypes.c_int)),
				("nbLabels",ctypes.c_int),
				("height",ctypes.c_long),
				("width",ctypes.c_long),
				("segmentation",np.ctypeslib.ndpointer(ctypes.c_longlong, flags="C_CONTIGUOUS")),
				("ske",np.ctypeslib.ndpointer(ctypes.c_short, flags="C_CONTIGUOUS")),
				("contours",np.ctypeslib.ndpointer(ctypes.c_long, flags="C_CONTIGUOUS")),
				("map",np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")),
				("labels",np.ctypeslib.ndpointer(ctypes.c_long, flags="C_CONTIGUOUS")),
				("pxC",ctypes.POINTER(ctypes.POINTER(ctypes.c_int))),
				("pxS",ctypes.POINTER(ctypes.POINTER(ctypes.c_int)))]




def initCudaCodes():
	so_file = "./SET/dmSET/gpudmSET.so" # 
	upIterationFunctions = ctypes.cdll.LoadLibrary(so_file)

	upIterationFunctions.initSETonGPU.restype = None
	upIterationFunctions.initSETonGPU.argtypes =[ctypes.POINTER(setImage),
	 ctypes.c_long,
	 np.ctypeslib.ndpointer(ctypes.c_longlong, flags="C_CONTIGUOUS"),
	 np.ctypeslib.ndpointer(ctypes.c_short, flags="C_CONTIGUOUS"),
	 np.ctypeslib.ndpointer(ctypes.c_long, flags="C_CONTIGUOUS"),
	 ctypes.c_long,
	 ctypes.c_long,
	ctypes.c_int]

	upIterationFunctions.freeMemory.restype = None
	upIterationFunctions.freeMemory.argtypes = [ctypes.POINTER(setImage)]#(struct setImage *set)


	upIterationFunctions.fullTesselation.restype = ctypes.c_void_p#None
	upIterationFunctions.fullTesselation.argtypes = [np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
	 np.ctypeslib.ndpointer(ctypes.c_long, flags="C_CONTIGUOUS"),
	 np.ctypeslib.ndpointer(ctypes.c_long, flags="C_CONTIGUOUS"), 
	 np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
	 np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
	 np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
	 np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
	 ctypes.POINTER(setImage)]

	return upIterationFunctions

def initCCodes():
	so_file = "./SET/dmSET/dmSET.so" 
	upIterationFunctions = ctypes.cdll.LoadLibrary(so_file) #.CDLL(so_file)

	upIterationFunctions.initSET.restype = ctypes.c_void_p
	upIterationFunctions.initSET.argtypes =[ctypes.POINTER(setImage),
	 ctypes.c_long,
	 np.ctypeslib.ndpointer(ctypes.c_longlong, flags="C_CONTIGUOUS"),
	 np.ctypeslib.ndpointer(ctypes.c_short, flags="C_CONTIGUOUS"),
	 np.ctypeslib.ndpointer(ctypes.c_long, flags="C_CONTIGUOUS"),
	 ctypes.c_long,
	 ctypes.c_long]

	upIterationFunctions.freeMemory.restype = None
	upIterationFunctions.freeMemory.argtypes = [ctypes.POINTER(setImage)]#(struct setImage *set)



	upIterationFunctions.fullTesselation.restype = ctypes.c_void_p
	upIterationFunctions.fullTesselation.argtypes = [np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
	 np.ctypeslib.ndpointer(ctypes.c_long, flags="C_CONTIGUOUS"),
	 ctypes.c_long,
	 np.ctypeslib.ndpointer(ctypes.c_long, flags="C_CONTIGUOUS"), 
	 np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
	 np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
	 np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
	 np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
	 np.ctypeslib.ndpointer(ctypes.c_longlong, flags="C_CONTIGUOUS"),
	 np.ctypeslib.ndpointer(ctypes.c_short, flags="C_CONTIGUOUS"),
	 np.ctypeslib.ndpointer(ctypes.c_long, flags="C_CONTIGUOUS"),
	 ctypes.c_long,
	 ctypes.c_long ,
	 ctypes.POINTER(setImage)]
	return upIterationFunctions

_labels = None
_border = None
_C = None
_rotangle = None
_labId = None
def _mean(labi_,):
	yxlabi = np.argwhere(_labels==_labId[labi_])
	if yxlabi.shape[0] <= 1 or _border[labi_]:
		return _C[labi_].copy(), _rotangle[labi_]
	else:
		Ci = np.mean(yxlabi, axis=0)
		try:
			#yxlabi[[1,0]] = yxlabi[[0,1]]
			yxcelli = yxlabi - Ci
			Sj = np.dot(yxcelli.T, yxcelli) / yxcelli.shape[0]
			lj, vj = np.linalg.eigh(Sj)
			alphaj = np.arctan2(vj[0,1], vj[1,1])
			
			if(np.abs(_rotangle[labi_]-alphaj) > np.pi / 2):
				if(alphaj<_rotangle[labi_]):
					alphaj += np.pi
				else:
					alphaj -= np.pi
			
		except np.linalg.LinAlgError as e:
			print('np.linalg.LinAlgError (k-means 2): %s' % e)
			return Ci, _rotangle[labi_]
		return Ci, alphaj

def mean_(labi_,labels_,border_,C_,rotangle_,labId_):
	yxlabi = np.argwhere(labels_==labId_[labi_])
	if yxlabi.shape[0] <= 1 or border_[labi_]:
		return C_[labi_].copy(), rotangle_[labi_]
	else:
		Ci = np.mean(yxlabi, axis=0)
		try:
			#yxlabi[[1,0]] = yxlabi[[0,1]]
			yxcelli = yxlabi - Ci
			Sj = np.dot(yxcelli.T, yxcelli) / yxcelli.shape[0]
			lj, vj = np.linalg.eigh(Sj)
			alphaj = np.arctan2(vj[0,1], vj[1,1])
			
			if(np.abs(rotangle_[labi_]-alphaj) > np.pi / 2):
				if(alphaj<rotangle_[labi_]):
					alphaj += np.pi
				else:
					alphaj -= np.pi
			
		except np.linalg.LinAlgError as e:
			print('np.linalg.LinAlgError (k-means 2): %s' % e)
			return Ci, rotangle_[labi_]
		return Ci, alphaj

def imageExtractCellFeatures(parameters,n_jobs):
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
	results = np.array(results)
	return results[:,0],results[:,1]


####################

#########################

def readOutputSimFile(fileName):

	f = open(fileName)
	data = f.readlines()
	f.close()
	init = []
	shuf = []
	out = []
	flag = 0
	for line in range(1,len(data)):
		l = data[line].split('\t')
		if l[0] == 'id_f' and line!=0 :
			flag = 2
		if l[0] == 'id_s' and line!=0 :
			flag = 1
		if flag ==0:
			init.append(data[line].split('\t'))
		elif  flag ==1 : 
			shuf.append(data[line].split('\t'))
		elif flag == 2 :
			out.append(data[line].split('\t'))
	init = np.array(init)
	out = np.array(out)
	shuf = np.array(shuf)
	return init[:,0].astype(int),init[:,1].astype(float),init[:,2].astype(float),init[:,3].astype(float),init[:,4].astype(float),init[:,5].astype(float),init[:,6].astype(float),init[:,7].astype(float),init[:,8].astype(float), shuf[1:,0].astype(int),shuf[1:,1].astype(float),shuf[1:,2].astype(float),shuf[1:,3].astype(float), out[1:,0].astype(int),out[1:,1].astype(float),out[1:,2].astype(float),out[1:,3].astype(float)


def dmSET(segFileName, maintOutputDirName, nbSimu = 1 , shuffle= None ,_interNumber = 500 , gpu = None,redoFile = None, shufpopInfo = None,dirmovie = None, pointFileName = None, n_jobs = 1) :

	if pointFileName :
		print('warping Not implemented with dmSET')
		sys.exit(0)

	if shuffle :
		nbSimu = nbSimu + 1

	for iSimu in range(nbSimu):
		## set output dir 
		if iSimu==0 :
			if nbSimu== 1 and redoFile : 
				nameDir = 'Redo'
				fOutfile =  redoFile[redoFile.rfind('/') : redoFile.rfind('.')] +'_re.csv'
			if nbSimu== 1 and shufpopInfo!= None : 
				nameDir = 'SubRand'
			if shuffle!= None :
				nameDir = 'Rand'
				fOutfile = ("/%04d"%iSimu)+'sLCSTable.tsv'
			else :
				nameDir = 'Rec'
				fOutfile = "/unshuffled"+'uLCSTable.tsv'
		else :
			nameDir = 'Rand'
			fOutfile = ("/%04d"%iSimu)+'sLCSTable.tsv'

		outputDirName = maintOutputDirName +'/'+nameDir+'SETDM_'+str(iSimu)
		if  not(os.path.exists(outputDirName)): 
			os.makedirs(outputDirName)
		else : 
			iF = 0
			while os.path.exists(outputDirName+'_'+str(iF)):
				iF = iF+1
			outputDirName= maintOutputDirName +'/'+nameDir+'SETDM_'+str(iSimu)+'_'+str(iF)		
			os.makedirs(outputDirName)
		fOutfile = outputDirName +fOutfile

		### init parameters from segmentation
		segFile = Image.open(segFileName)
		segImg = np.array(segFile).astype(np.int64)
		segFile.close()

		height,width = segImg.shape

		labelsId_raw = np.unique(segImg)
		labelsId = np.arange(0,len(labelsId_raw))
		rid = np.zeros(np.max(labelsId_raw) + 1, dtype=int)
		rid[labelsId_raw] = np.arange(labelsId_raw.shape[0])#reverse ids = labelsId

		segImg = rid[segImg]


		LabelsInit = segImg.reshape((height*width,))
		I = np.array(list(itertools.product(range(height),range(width))))
		C_init = [np.mean(I[np.where(LabelsInit==labi)], axis=0) for labi in labelsId]
		rotangle_init = np.zeros((len(labelsId),), dtype=np.float)
		labels = None #np.zeros((height*width,))


		bordersLabels = np.unique(np.concatenate([segImg[:,0],segImg[:,width-1],segImg[0,:],segImg[height-1,:]]))
		borders =  np.zeros((len(labelsId),1), dtype=bool)
		borders[np.isin(labelsId, bordersLabels) ] =1

		global _labels,_border,_C,_rotangle,_labId
		_labels = segImg
		_border = borders
		_C = C_init
		_rotangle = rotangle_init
		_labId = labelsId
		parameters = range(len(labelsId))
		C,rotangle_init = imageExtractCellFeatures(parameters, n_jobs) #np.array([mean_(labi,segImg,borders,C_init,rotangle_init,labelsId) for labi in range(len(labelsId))]).T 

		rotangle_init = rotangle_init.astype(np.float64)
		rotangle = np.copy(rotangle_init)
		C = np.array(list(C),dtype = np.float64)
		C_init = np.array(C_init,dtype = np.float64)

		#save parameters
		simFeaturesFile = open(fOutfile,'w')
		simFeaturesFile.write('id\tCx\tCy\ttheta\tl1\tl2\ta1\ta2\tp\n')
		for ilabel in range(len(labelsId_raw)):
			simFeaturesFile.write(str(labelsId_raw[ilabel]) + '\t' +str(C[ilabel][0]) +'\t' +str(C[ilabel][1])+'\t'+ str(rotangle[ilabel]) +'\t' +str('')+'\t'+str('') +'\t' +str('')+'\t'+str('')+'\t'+str('')+'\n')
		simFeaturesFile.close()

		## position modifications
		if shuffle :
			nbidx = np.where(~np.array(borders))[0]
			Inotborder = I[np.isin(LabelsInit,labelsId[nbidx])]
			C[nbidx] = list(map(np.array,Inotborder[np.random.choice(np.arange(Inotborder.shape[0]), nbidx.shape[0], replace=False)]))
			rotangle[nbidx] = np.random.uniform(-np.pi, np.pi, nbidx.shape[0])

		if shufpopInfo!= None :
			labelClass,cellClass = readClassOfSubpop(shufpopInfo[0])
			cellsToShuf=[]
			for popToshuffle in range(1,len(shufpopInfo)):
				cellsToShuf.append( np.squeeze(labelClass[np.argwhere(cellClass==str(shufpopInfo[popToshuffle]))]) )
			cellsToShuf = np.concatenate(cellsToShuf)
			boolCells = [ilabel in cellsToShuf for ilabel in labelsId]
			nbidx = np.where(~np.array(borders) & boolCells)[0]

			Inotborder = I[np.isin(LabelsInit,labelsId[nbidx])]
			C[nbidx] = np.array(list(map(np.array,Inotborder[np.random.choice(np.arange(Inotborder.shape[0]), nbidx.shape[0], replace=False)])),dtype = np.float64)
			rotangle[nbidx] = np.random.uniform(-np.pi, np.pi, nbidx.shape[0])

		if redoFile : 
			if shuffle :
				print('-s and -r options are not compatible')
				print('end')
				sys.exit(0)

			id,_,_,_,_,_,_,_,_,_,Cx,Cy,rotangle,_,_,_,_  = readOutputSimFile(redoFile)
			C = np.array([Cx,Cy]).T

		#save parameters after shuffle
		simFeaturesFile = open(fOutfile,'a')
		simFeaturesFile.write('id_s\tCx_s\tCy_s\ttheta_s\n')
		for ilabel in range(len(labelsId_raw)): 
			simFeaturesFile.write(str(labelsId_raw[ilabel]) + '\t' +str(C[ilabel][0]) +'\t' +str(C[ilabel][1])+'\t'+ str(rotangle[ilabel]) +'\n')
		simFeaturesFile.close()


		##cells borders
		allborder = morphological_gradient(segImg, size=3) #> 0)
		#plt.imshow(allborder)
		#plt.show()
		imgBorders = np.ones((height,width),dtype = np.int64)*-1
		imgBorders[allborder>0] = segImg[allborder>0]
		#plt.imshow(imgBorders)
		#plt.show()
		minNormalize = np.inf
		segImg[imgBorders!=-1] = -1

		#initialize skeleton 
		erodSeg = np.zeros((height,width))
		erodSeg[segImg!=-1] = 1
		if 1 : 
			newSke = skeletonize(erodSeg).astype(np.int16)
		else : 
			erodSeg = medfilt (erodSeg, 3)
			medAxis= medial_axis(erodSeg)

			noInsideLabels = np.zeros((len(labelsId,)))

			newSke = np.zeros((height,width),dtype = np.int16)
			for eachLab in range(len(labelsId)):

				imgBordersLabi = np.ones((height,width))
				imgBordersLabi[imgBorders==labelsId[eachLab]] = 0
				distanceMap = distance_transform_edt(imgBordersLabi)
				yxSke = np.argwhere((medAxis==True) & (segImg == labelsId[eachLab]))	
				if len(yxSke) == 0 :
					print('label '+ str(labelsId[eachLab])+' no ske at all')

				### construct graph
				distSke = cdist(yxSke,yxSke)	
				distSke[distSke >= 2] = 0
				G = nx.from_numpy_matrix(distSke)

				if len(G.nodes()) <1:
					print('label '+ str(labelsId[eachLab])+' no ske at all : took a random pixel in erroded segmentation')
					yx =  np.argwhere(segImg == labelsId[eachLab])
					if len(yx) == 0:
						noInsideLabels[eachLab] = 1

						print('label '+ str(labelsId[eachLab])+' took a all pixel in border')
						yx = np.argwhere(imgBorders==labelsId[eachLab])
						newSke[yx[:,0],yx[:,1]] = 1
					else :
						randPix = np.random.randint(len(yx))
						newSke[yx[randPix][0],yx[randPix][1]] = 1
				else : 
					pos = {i : (yxSke[i,0],yxSke[i,1]) for i in G.nodes()} #dict([[i,(yxSke[i,0],yxSke[i,1])] for i in G.nodes()])


					labelsG = {i : distanceMap[yxSke[i,0],yxSke[i,1]] for i in G.nodes()}
					connex = dict(nx.degree(G))
					extrem = np.where(np.array(list(connex.values()))==1)
					idxExtrem = np.array(list(connex.keys()))[extrem]
					## grad in graph
					gradG = {}
					gradVals= []
					for eachNode in G.nodes():
						neig_n = list(G[eachNode].keys())
						gradNod = {}
						for eachNeig in neig_n :
							valnn = (labelsG[eachNode]-labelsG[eachNeig])/np.sqrt(np.power(pos[eachNode][0]- pos[eachNeig][0],2) +np.power(pos[eachNode][1]-pos[eachNeig][1],2))
							gradVals.append(valnn)
							gradNod[eachNeig] = valnn
						gradG[eachNode] = gradNod
	
					flag = 1
					while flag :
						if len(extrem)==0 or len(G.nodes()) <2:
							break
						stdValuePerExtrem = []
						muValuePerExtrem = []
						r_skExtrem = []
						gradExtrem = []
						for eachExtrem in range(len(idxExtrem)):
							edi = idxExtrem[eachExtrem]
							gradExtrem.append(gradG[edi][list(G[edi].keys())[0]])
			
						EToRemove = np.array(gradExtrem)<=0
						if EToRemove.sum() == 0  or (len(G.nodes())-EToRemove.sum())<1:
							flag =0 #exit while
						else :
		
							G.remove_nodes_from(idxExtrem[EToRemove])
							connex = dict(nx.degree(G))
							extrem = np.where(np.array(list(connex.values()))==1)
							idxExtrem = np.array(list(connex.keys()))[extrem]
					for eachNode in G.nodes():
						newSke[pos[eachNode][0],pos[eachNode][1]] = 1

		medAxis = newSke
		if dirmovie:
			Image.fromarray(newSke).save(outputDirName + '/rawSke.tif') 
			Image.fromarray(segImg.astype(np.int16)).save(outputDirName + '/segImg.tif') 
		cycle = 0
		C = np.concatenate(C).reshape((C.shape[0],2)).astype(np.float64)
		rotangle = np.array(list(rotangle_init), dtype = np.float64)

		eMove = []
		print('save in : '+outputDirName)
		np.save(outputDirName+'/CInit.npy',C)
		np.save(outputDirName+'/RotangleInit.npy',rotangle)
		Image.fromarray(newSke.astype(np.int8)).save(outputDirName+'/skeletons.tif')

		if gpu!=None : ######################## gpu ########################
			t1 = time.time()
			upIterationFunctions = initCudaCodes()
			structGPU = setImage()
			upIterationFunctions.initSETonGPU(structGPU, len(labelsId), segImg, medAxis, imgBorders, width, height,gpu)
			print('time of init on gpu: ',time.time() -t1)
			while (cycle<_interNumber) :
				print(cycle)
				t1 = time.time()

				labels = np.zeros((height,width),dtype = np.int64)
				distanceMapFull = np.ones((height,width),dtype = np.float64)*np.inf
		
				oldC  = np.copy(np.concatenate(C).reshape((C.shape[0],2)))
				upIterationFunctions.fullTesselation( distanceMapFull, labels, labelsId, -1*rotangle, C, -1*rotangle_init, C_init, structGPU)
				print('time iter gpu: ',time.time() -t1)

				t3 = time.time()

				_labels = labels
				_C = C
				_rotangle = rotangle
				C,rotangle = imageExtractCellFeatures(parameters, n_jobs) #np.array([mean_(labi,labels,borders,C,rotangle,labelsId) for labi in range(len(labelsId))]).T			

				t4 = time.time()
				print('time mean : ', t4-t3)
				cycle += 1

				C = np.concatenate(C).reshape((C.shape[0],2)).astype(np.float64)
				rotangle = np.array(list(rotangle), dtype = np.float64)

				deltaC = np.sqrt(np.power(C[:,0]-oldC[:,0],2)+np.power(C[:,1]-oldC[:,1],2))
				eMove.append(deltaC.sum())
				print('deltaC : '+str(deltaC.sum()) +' (nbC qui bougent :'+str((~borders).sum())+')')
				if dirmovie:
					Image.fromarray(distanceMapFull.astype(np.longdouble).astype(np.float64)).save('%s/mapFull_%04d_%s.tif' % (outputDirName, cycle, len(labelsId)))
					Image.fromarray(labelsId_raw[labels].astype(np.long).astype(np.int32)).save('%s/iteration_%04d_%s.png' % (outputDirName, cycle, len(labelsId)))

			np.save(outputDirName+'/CFinal.npy',C)
			np.save(outputDirName+'/RotangleFinal.npy',rotangle)
			np.save(outputDirName+'/eMove.npy',eMove)
			Image.fromarray(distanceMapFull.astype(np.longdouble).astype(np.float64)).save('%s/mapFull_%04d_%s.tif' % (outputDirName, cycle, len(labelsId)))
			Image.fromarray(labelsId_raw[labels].astype(np.long).astype(np.int32)).save('%s/iteration_%04d_%s.png' % (outputDirName, cycle, len(labelsId)))
			upIterationFunctions.freeMemory(structGPU)

		else :  ################## CPU ###############################
			upIterationFunctions = initCCodes()
			structSET = setImage()
			upIterationFunctions.initSET(structSET, len(labelsId), segImg, medAxis, imgBorders, width, height)
			#reconstruct imgC from structSET
			imgStructC = np.zeros((height,width),dtype = np.int32)
			for eachL in range(len(labelsId)):
				for eachPx in range(structSET.nbPxS[eachL]):
					imgStructC[structSET.pxS[eachL][eachPx*2],structSET.pxS[eachL][eachPx*2+1]]=eachL
			Image.fromarray(imgStructC).save('%s/reconstructedCFromStruct_%04d_%s.tif' % (outputDirName, cycle, len(labelsId)))
			while cycle<_interNumber:
				print(cycle)
				t1 = time.time()
				labels = np.zeros((height,width),dtype = np.int64)
				distanceMapFull = np.ones((height,width),dtype = np.float64)*np.inf
				oldC  = np.copy(np.concatenate(C).reshape((C.shape[0],2)))

				rotangle = np.array(list(rotangle), dtype = np.float64)
				C = np.array(list(C),dtype = np.float64)
				#
				upIterationFunctions.fullTesselation( distanceMapFull, labels, len(labelsId), labelsId, -1*rotangle, C, -1*rotangle_init, C_init, segImg, medAxis, imgBorders, width, height,structSET)
				t2 = time.time()
				print('time iter : ', t2-t1)

				Image.fromarray(distanceMapFull.astype(np.longdouble).astype(np.float64)).save('%s/mapFull_%04d_%s.tif' % (outputDirName, cycle, len(labelsId)))
				Image.fromarray(labelsId_raw[labels].astype(np.long).astype(np.int32)).save('%s/iteration_%04d_%s.png' % (outputDirName, cycle, len(labelsId)))
				t3 = time.time()
				_labels = labels
				_C = C
				_rotangle = rotangle
				C,rotangle = imageExtractCellFeatures(parameters, n_jobs)# np.array([mean_(labi,labels,borders,C,rotangle,labelsId) for labi in range(len(labelsId))]).T

				C = np.concatenate(C).reshape((C.shape[0],2)).astype(np.float64)
				deltaC = np.sqrt(np.power(C[:,0]-oldC[:,0],2)+np.power(C[:,1]-oldC[:,1],2))
				eMove.append(deltaC.sum())


				t4 = time.time()
				print('time mean : ', t4-t3)
				if dirmovie:
					Image.fromarray(distanceMapFull.astype(np.longdouble).astype(np.float64)).save('%s/mapFull_%04d_%s.tif' % (outputDirName, cycle, len(labelsId)))
					Image.fromarray(labelsId_raw[labels].astype(np.long).astype(np.int32)).save('%s/iteration_%04d_%s.png' % (outputDirName, cycle, len(labelsId)))
				cycle += 1
			np.save(outputDirName+'/eMove.npy',eMove)
			Image.fromarray(distanceMapFull.astype(np.longdouble).astype(np.float64)).save('%s/mapFull_%04d_%s.tif' % (outputDirName, cycle, len(labelsId)))
			Image.fromarray(labelsId_raw[labels].astype(np.long).astype(np.int32)).save('%s/iteration_%04d_%s.png' % (outputDirName, cycle, len(labelsId)))
			upIterationFunctions.freeMemory(structSET)

		# add simulation shape information to the file
		simFeaturesFile = open(fOutfile,'a')
		simFeaturesFile.write('id_f\tCx_f\tCy_f\ttheta_f\n')
		for ilabel in range(len(labelsId_raw)): 
			simFeaturesFile.write(str(labelsId_raw[ilabel]) + '\t' +str(C[ilabel][0]) +'\t' +str(C[ilabel][1])+'\t'+ str(rotangle[ilabel]) +'\n')
		simFeaturesFile.close()

