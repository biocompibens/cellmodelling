# analyze P30 number of contact between two stem cells. 
import matplotlib as mpl
mpl.use('Agg')#('TkAgg') #
import glob

import numpy as np
import os 
import skimage.future
import scipy
from scipy.spatial.distance import cdist
from PIL import Image

import matplotlib.pyplot as plt
from IPython import embed

simulation = ['./dataExamples/ependymaP30/reconstruction/unshuffled.tiff']

directory = './dataExamples/ependymaP30/allSimuOutput/'
outputRep = './dataExamples/ependymaP30/nbContactAnalysis'

if not(os.path.exists(outputRep)): 
	os.makedirs(outputRep)


couleurShuffle = (96/255.,26/255.,162/255.) 
couleurUnshuffle = (159/255.,41/255.,94/255.) 

def label2Contours(ImgL1):
	region_borders = scipy.ndimage.morphological_gradient(ImgL1,
                                            footprint=[[0, 1, 0],
                                                       [1, 1, 1],
                                                       [0, 1, 0]])
	yxB = np.argwhere(region_borders!=0)
	region_borders[yxB[:,0],yxB[:,1]]=1
	return region_borders 

distriObs = []
syntheticContacts = []
for eachSynthetic in xrange(len(simulation)):
	plt.clf()
	nameImg = simulation[eachSynthetic][simulation[eachSynthetic].rfind('/')+1:simulation[eachSynthetic].rfind('.')]
	print nameImg
	patternFile = Image.open(simulation[eachSynthetic])
	imgPattern = np.array(patternFile).astype(np.int)
	patternFile.close()
	if eachSynthetic==1:
		imgPattern = imgPattern-1
	labels = np.unique(imgPattern)
	imC = label2Contours(imgPattern)

	if 1 : 
		classFileName = './dataExamples/ependymaP30/P30_424_9_cells_features.csv'
		fc = open(classFileName,'r')
		lines = fc.readlines()
		fc.close()

		listeCelluleSouche=[]
		imCluster= np.zeros(imC.shape,np.float16)
		for iline in xrange(1,len(lines)):
			lines[iline]=lines[iline].split(',')
			if lines[iline][2][:-1]=='stem':
				yx = np.argwhere(imgPattern==int(lines[iline][1]))
				listeCelluleSouche.append(int(lines[iline][1]))
				imCluster[yx[:,0],yx[:,1]]=0.5
		imCluster[imC==1]=1
		Image.fromarray(imCluster.astype(np.float)).save(outputRep+'/reconstructionClass.tif')

	ragL = skimage.future.graph.rag_boundary(imgPattern, imC)
	keysRag= np.array(ragL.edges.keys())
	

	positifCells = listeCelluleSouche

	nbContactUnshuf = np.isin(np.argwhere(np.isin(keysRag[:,0],positifCells)),np.argwhere(np.isin(keysRag[:,1],positifCells))).sum()
	syntheticContacts.append(nbContactUnshuf)

#shuffle Analysis
distriContactShuffled = []
shuffleTissues = glob.glob(directory+'/[0-9]*.tiff')
for eachShuf in xrange(len(shuffleTissues)):	

	patternFile = Image.open(shuffleTissues[eachShuf])
	imgPattern = np.array(patternFile).astype(np.int)
	patternFile.close()

	labels = np.unique(imgPattern)

	imC = label2Contours(imgPattern)
	if eachShuf==1:
		imCluster= np.zeros(imC.shape,np.float16)
		for lab in labels:
			if lab in listeCelluleSouche:
				yx = np.argwhere(imgPattern==lab)
				imCluster[yx[:,0],yx[:,1]]=0.5
		imCluster[imC==1]=1
		Image.fromarray(imCluster.astype(np.float)).save(outputRep+'/shuff'+shuffleTissues[eachShuf][shuffleTissues[eachShuf].rfind('/')+1:shuffleTissues[eachShuf].rfind('.')]+'Class.tif')


	ragL = skimage.future.graph.rag_boundary(imgPattern, imC)
	keysRag= np.array(ragL.edges.keys())

	nbContactUnshuf = np.isin(np.argwhere(np.isin(keysRag[:,0],positifCells)),np.argwhere(np.isin(keysRag[:,1],positifCells))).sum()
	distriContactShuffled.append(nbContactUnshuf)

np.save( outputRep+'/nbContactShuffle_stemCells',distriContactShuffled) 


plt.clf()
fig = plt.figure(figsize = [15,10])
plt.hist(distriContactShuffled, color =couleurShuffle, histtype='step',label = 'random SET')
plt.xlabel('number of contact between two stem cells')
plt.ylabel('Image quantity')
plt.vlines(syntheticContacts[0],0,100, color = (0,0,0), linestyles = 'solid', label = 'reconstruction by SET',lw= 3)
plt.legend()

lenShuff = float(len(distriContactShuffled))
proba=[]
for eachCase in xrange(1):
	if syntheticContacts[eachCase]>np.mean(distriContactShuffled) :
		proba.append(len(np.argwhere(np.array(distriContactShuffled)>=syntheticContacts[eachCase]))/lenShuff)
	else :
		proba.append(len(np.argwhere(np.array(distriContactShuffled)<=syntheticContacts[eachCase]))/lenShuff)
	if proba[-1] == 0 :
		proba[-1] = 1./(lenShuff+1)
	plt.text(syntheticContacts[eachCase], lenShuff,'%.3f'%(proba[eachCase]),color = couleurUnshuffle)

plt.ylim(ymax= lenShuff)

plt.savefig(outputRep+'/nbContactObsvsShuf.pdf') 
plt.close()


