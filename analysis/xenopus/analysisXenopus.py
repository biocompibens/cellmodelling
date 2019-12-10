## code to analyze cell organization of the xenopus mucociliary epidermis
## need to have generate a random SET before with : 
## python ./SET/model.py -l ./dataExamples/xenopus_St33/20x-st33-ctrl-3_SME_Projection_CH1_refCH2_numCH2_UCH0_DCH2_CF-catchment-basins_wdams.tif -o ./dataExamples/xenopus_St33/allSimuOutput_INC -s 1000  -p 8 -sp ./dataExamples/xenopus_St33/20x-st33-ctrl-3Class.npy SSC INC MCC
## or 
## python ./SET/model.py -l ./dataExamples/xenopus_St33/20x-st33-ctrl-3_SME_Projection_CH1_refCH2_numCH2_UCH0_DCH2_CF-catchment-basins_wdams.tif -o ./dataExamples/xenopus_St33/allSimuOutput_SSC -s 1000  -p 8 -sp ./dataExamples/xenopus_St33/20x-st33-ctrl-3Class.npy SSC 

import glob
from PIL import Image
import numpy as np
import matplotlib as mpl
mpl.use('Agg')#mpl.use('TkAgg')#
import matplotlib.pyplot as plt
import glob
import skimage.future
import scipy.ndimage

def label2Contours(ImgL1):
	region_borders = scipy.ndimage.morphological_gradient(ImgL1,
                                            footprint=[[0, 1, 0],
                                                       [1, 1, 1],
                                                       [0, 1, 0]])
	yxB = np.argwhere(region_borders!=0)
	region_borders[yxB[:,0],yxB[:,1]]=1
	return region_borders 

def calculPvalue(obs,shuffleDistri,valuesHistoBins, binLimits, name):

	if obs<binLimits[0]:
		print "Proba of having this (small) number of contact between "+name+" by chance is not obtain with " +str(len(shuffleDistri))+" replicats"
	elif obs>binLimits[len(binsLim)-1]:
		print "Proba of having this (high) number of contact between "+name+" by chance is not obtain with " +str(len(shuffleDistri))+" replicats"
	else :
		meanVal = np.mean(shuffleDistri)

		if obs> meanVal :
			nbObs = len(np.where(np.array(shuffleDistri)>=obs)[0])
			proba = float(nbObs)/valuesHistoBins.sum()
			orientation = "high"
		else : 
			
			nbObs = len(np.where(np.array(shuffleDistri)<=obs)[0])
			proba = float(nbObs)/valuesHistoBins.sum()
			orientation = "small"
		print "Probability of having this ("+orientation+") number of contact between "+name+" by chance = " +str(proba)
		
	return

def analysisXenopus(dirData,dirSimulation) :

	classificationFile= dirData+'/20x-st33-ctrl-3Class.npy'
	segmentationImage = dirData+'/20x-st33-ctrl-3_SME_Projection_CH1_refCH2_numCH2_UCH0_DCH2_CF-catchment-basins_wdams.tif'

	patternFile = Image.open(segmentationImage)
	imgSeg = np.array(patternFile).astype(np.int)
	patternFile.close()
	labels = np.unique(imgSeg)

	#load cell classification
	classification = np.load(classificationFile)
	j = np.squeeze(np.argwhere(np.array(classification) == 'ISC'))
	v = np.squeeze(np.argwhere(np.array(classification) == 'MCC'))
	r = np.squeeze(np.argwhere(np.array(classification) == 'SSC'))
	b = np.squeeze(np.argwhere(np.array(classification) == 'Goblet'))
	j = labels[j]
	v = labels[v]
	r = labels[r]
	b = labels[b]

	unshuffleSimulationFile  = glob.glob(dirSimulation+'/unshuffled.tiff')
	shuffleSimulationsFiles = glob.glob(dirSimulation +'/[0-9]*.tiff')


	## analysis of unshuffle 
	patternFile = Image.open(unshuffleSimulationFile[0] )
	unshuffleSimulation= np.array(patternFile).astype(np.int)
	patternFile.close()
	labels= np.unique(unshuffleSimulation)
	unshuffleSimulationC = label2Contours(unshuffleSimulation)
	ragL = skimage.future.graph.rag_boundary(unshuffleSimulation, unshuffleSimulationC)
	keysRag= np.array(ragL.edges.keys())


	linkVVunshuffle = 0
	linkRVunshuffle = 0

	for eachLink in xrange(len(keysRag)):
		if keysRag[eachLink][0] in v and keysRag[eachLink][1] in v:
			linkVVunshuffle +=1
		elif (keysRag[eachLink][0] in r and keysRag[eachLink][1] in v) or (keysRag[eachLink][1] in r and keysRag[eachLink][0] in v):
			linkRVunshuffle +=1

	## anlyse pour shuffles - null distribution building
	linkVVshuffle = []
	linkRVshuffle = []

	for each in xrange(len(shuffleSimulationsFiles)) : 
		patternFile = Image.open(shuffleSimulationsFiles[each] )
		shuffleSimulation= np.array(patternFile).astype(np.int)
		patternFile.close()
		labels= np.unique(shuffleSimulation)
		shuffleSimulationC = label2Contours(shuffleSimulation)
		ragL = skimage.future.graph.rag_boundary(shuffleSimulation, shuffleSimulationC)
		keysRag= np.array(ragL.edges.keys())

		linkVV = 0
		linkRV = 0


		for eachLink in xrange(len(keysRag)):
			if keysRag[eachLink][0] in v and keysRag[eachLink][1] in v:
				linkVV +=1
			elif (keysRag[eachLink][0] in r and keysRag[eachLink][1] in v) or (keysRag[eachLink][1] in r and keysRag[eachLink][0] in v):
				linkRV +=1


		linkVVshuffle.append(linkVV)
		linkRVshuffle.append(linkRV)

	# display results (distribution vs unshuffle)
	couleurShuffle = (96/255.,26/255.,162/255.) #(0.,0.6,0.8)
	couleurUnshuffle = (255/255.,88/255.,0/255.) #(159/255.,41/255.,94/255.) #'g'
	outputDir = './'

	plt.clf()
	try :
		valuesBins,binsLim, patches = plt.hist(linkVVshuffle,bins = np.max(linkVVshuffle)-np.min(linkVVshuffle) ,histtype='step', color =  couleurShuffle, align = 'left',lw = 2) #normed=True,
	except :
		valuesBins,binsLim, patches = plt.hist(linkVVshuffle,histtype='step', color =  couleurShuffle, align = 'left',lw = 2) #normed=True,
	plt.vlines(linkVVunshuffle,0,50,color= couleurUnshuffle)
	plt.title('link V-V')
	plt.xlabel('number of connected MCC')
	plt.ylabel('quantity')
	plt.savefig(outputDir +'/linkVV.pdf')

	calculPvalue(linkVVunshuffle,linkVVshuffle,valuesBins, binsLim, "between MCC")

	plt.clf()
	try :
		valuesBins,binsLim, patches = plt.hist(linkRVshuffle,bins = np.max(linkRVshuffle)-np.min(linkRVshuffle) ,histtype='step', color =  couleurShuffle, align = 'left',lw = 2)
	except: 
		valuesBins,binsLim, patches = plt.hist(linkRVshuffle,histtype='step', color =  couleurShuffle, align = 'left',lw = 2)
	plt.vlines(linkRVunshuffle,0,50,color= couleurUnshuffle)
	plt.title('link R-V')
	plt.ylabel('quantity')
	plt.xlabel('number of SSC connected to MCC')
	plt.xticks(range(0,np.max(linkRVshuffle)))
	plt.savefig(outputDir +'/linkRV.pdf')
	calculPvalue(linkRVunshuffle,linkRVshuffle,valuesBins, binsLim, "between SSC and MCC")

	plt.clf()
	plt.figure(figsize = (6,5))
	valuesBins,binsLim, patches = plt.hist(linkRRshuffle ,bins = np.max(linkRRshuffle)-np.min(linkRRshuffle),histtype='step', color =  couleurShuffle, align = 'left',lw = 2)
	plt.vlines(linkRRunshuffle,0,40,color= couleurUnshuffle)
	plt.title('link R-R')
	plt.ylabel('quantity')
	plt.xlabel('number of connected SSC')
	plt.xlim(xmin=-2,xmax=12)
	plt.ylim(ymax=70)
	plt.savefig(outputDir +'/linkRR.pdf')

	calculPvalue(linkRRunshuffle,linkRRshuffle,valuesBins, binsLim, "between SSC")

dirXenopus ='./dataExamples/xenopus_St33'
dirSimulation = './dataExamples/xenopus_St33/allSimuOutput_INC' # to change allSimuOutput_SSC
analysisXenopus(dirXenopus, dirSimulation)
