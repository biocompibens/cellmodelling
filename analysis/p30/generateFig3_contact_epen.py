##code pour figure 3 ! contact
from random import shuffle,sample
from skimage.measure import label, regionprops
import random
import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import skimage.future.graph
from scipy.spatial.distance import cdist
from PIL import ImageFilter

### needed functions


def extractObservedNumberOfContact(FileName,listeCelluleSouche) :
	patternFile = Image.open(FileName)
	imgPattern = np.array(patternFile).astype(np.int)
	patternFile.close()

	labels = np.unique(imgPattern)
	imC = label2Contours(imgPattern)
		
	if 1:
		pilImC= Image.fromarray(imC.astype(np.int8))
		pilImC = pilImC.filter(ImageFilter.MaxFilter(3))
		pilImC = pilImC.filter(ImageFilter.MaxFilter(3))
		imC = np.array(pilImC)
		imCluster= np.zeros(imC.shape,np.float16)
		for lab in labels:
			if lab in listeCelluleSouche:
				yx = np.argwhere(imgPattern==lab)
				imCluster[yx[:,0],yx[:,1]]=0.5
		imCluster[imC==1]=1
		Image.fromarray(imCluster.astype(np.float)).save(outputRep+'/unshuff'+FileName[FileName.rfind('/')+1:FileName.rfind('.')]+'Class.tif')
		print outputRep+'unshuff'+FileName[FileName.rfind('/')+1:FileName.rfind('.')]+'Class.tif'


	print "Number of cell :" +str(len(labels))+", Number of stem cell = "+str(len(listeCelluleSouche))+ ", % of stem cell :"+str(len(listeCelluleSouche)*100./len(labels))

	## Analyse simu/obs
	ragL = skimage.future.graph.rag_boundary(imgPattern, imC)
	keysRag= np.array(ragL.edges.keys())
	nbContactUnshuf = np.isin(np.argwhere(np.isin(keysRag[:,0],listeCelluleSouche)),np.argwhere(np.isin(keysRag[:,1],listeCelluleSouche))).sum()

	return nbContactUnshuf

def shuffleLabel(FileName,listeCelluleSouche) : 
	patternFile = Image.open(FileName)
	imgPattern = np.array(patternFile).astype(np.int)
	patternFile.close()

	labels = np.unique(imgPattern)
	borderLabels = np.unique(np.concatenate([np.unique(imgPattern[0,:]), np.unique(imgPattern[-1,:]), np.unique(imgPattern[:,0]), np.unique(imgPattern[:,-1])]))
	labels = np.array([ i for i in labels if not(i in borderLabels)])

	#boundaries = np.unique(np.concatenate((np.unique(imgPattern[1,:]), np.unique(imgPattern[-1,:]), np.unique(imgPattern[:,-1]), np.unique(imgPattern[:,1]))) )

	imC = label2Contours(imgPattern)
	ragL = skimage.future.graph.rag_boundary(imgPattern, imC)
	keysRag= np.array(ragL.edges.keys())
	nbContactShuffle = []
	for ishuffle in xrange(1000):
		if 0 :
			slist=np.copy(labels)#range(len(labels))
			shuffle(slist) #new index in the list if slist[0]== 4 : label[0] = label[4]
			slist = np.array(slist)

			shuffledListeCelluleSouche = list(slist[np.array(listeCelluleSouche)-1])
		elif 0 : 
			slist=sample(labels, len(labels))
			shuffledListeCelluleSouche = list(slist[np.array(listeCelluleSouche)-1])
		else : 
			shuffledListeCelluleSouche =np.random.choice(labels,len(listeCelluleSouche),replace = False)
		nbContactShuf = np.isin(np.argwhere(np.isin(keysRag[:,0],shuffledListeCelluleSouche)),np.argwhere(np.isin(keysRag[:,1],shuffledListeCelluleSouche))).sum()
		nbContactShuffle.append(nbContactShuf)
		if ishuffle in [0] :
			pilImC= Image.fromarray(imC.astype(np.int8))
			pilImC = pilImC.filter(ImageFilter.MaxFilter(3))
			imC = np.array(pilImC)
			exShuf = np.zeros(imC.shape)
			exShuf[np.isin(imgPattern,shuffledListeCelluleSouche)]=2
			exShuf[imC ==1] = 1
			Image.fromarray(exShuf.astype(np.uint32)).save(FileName[:FileName.rfind('.')]+'_aShufflesID'+str(ishuffle)+'NbContact'+str(nbContactShuf)+'.tif')

	return 	nbContactShuffle 


### grid
def gridShuffling(imgSimuFileName,listeCelluleSouche):
	gridImageFileName = '/users/biocomp/laruelle/coupling/Syn.png'

	gridImageFile = Image.open(gridImageFileName)
	imgGrid = np.array(gridImageFile)[:,230:1520,0]
	gridImageFile.close()

	MarkersImg = np.copy(imgGrid)
	MarkersImg[MarkersImg!=255]=0
	cnnp = scipy.ndimage.label(MarkersImg)
	labeledImg = skimage.segmentation.watershed(imgGrid, cnnp[0])

	smallgrid = np.copy(imgGrid[:712,:712])
	MarkersImg = np.copy(smallgrid)
	MarkersImg[MarkersImg!=0]=255
	cnnp = scipy.ndimage.label(MarkersImg)

	borders = np.argwhere(cnnp[0]==0)
	p = regionprops(cnnp[0])
	c = [p[i].centroid for i in xrange(cnnp[1])]
	nearestLabelOfBorders = np.argmin(cdist(borders,c),axis=1)
	labeledImg=np.copy(cnnp[0])
	labeledImg[borders[:,0],borders[:,1]]=nearestLabelOfBorders+1


	labeledGrid = np.copy(labeledImg[100:-100,100:-100])
	labelsGrid = np.unique(labeledGrid)
	labelsGridLinear = np.copy(labeledGrid)
	for i in xrange(len(labelsGrid)):
		yx= np.argwhere(labeledGrid == labelsGrid[i])
		labelsGridLinear[yx[:,0],yx[:,1]]=i+1
	labeledGrid = np.copy(labelsGridLinear)
	labelsGrid = np.unique(labeledGrid)
	borderLabelsGrid = np.unique(np.concatenate([np.unique(labeledGrid[0,:]), np.unique(labeledGrid[-1,:]), np.unique(labeledGrid[:,0]), np.unique(labeledGrid[:,-1])]))
	labelsGrid = np.array([ i for i in labelsGrid if not(i in borderLabelsGrid)])

	print "number of label : " +str(len(labelsGrid))
	#Image.fromarray(labeledGrid.astype(np.uint32)).save(imgSimuFileName[:imgSimuFileName.rfind('.')]+'_correspondingGrid.tif')
	labeledGridC = label2Contours(labeledGrid)
	ragL = skimage.future.graph.rag_boundary(labeledGrid, labeledGridC)
	keysRag= np.array(ragL.edges.keys())

	patternFile = Image.open(imgSimuFileName)
	imgPattern = np.array(patternFile).astype(np.int)
	patternFile.close()

	labels = np.unique(imgPattern)
	borderLabels = np.unique(np.concatenate([np.unique(imgPattern[0,:]), np.unique(imgPattern[-1,:]), np.unique(imgPattern[:,0]), np.unique(imgPattern[:,-1])]))
	labels = np.array([ i for i in labels if not(i in borderLabels)])
	print "number of label not at the border in Grid : " +str(len(labelsGrid))+' and in the corresponding img :' +str(len(labels))


	nbContactShuffle = []
	for ishuffle in xrange(1000):

		if 0 : 
			slist=np.copy(labelsGrid)
			shuffle(slist) 
			slist = np.array(slist)

			shuffledListeCelluleSouche = list(slist[np.array(listeCelluleSouche)-1])
		elif 0 : 
			slist=np.array(sample(labelsGrid, len(labelsGrid)))
			shuffledListeCelluleSouche = list(slist[np.array(listeCelluleSouche)-1])
		else : 
			shuffledListeCelluleSouche =np.random.choice(labelsGrid,len(listeCelluleSouche),replace = False)
		nbContactShuf = np.isin(np.argwhere(np.isin(keysRag[:,0],shuffledListeCelluleSouche)),np.argwhere(np.isin(keysRag[:,1],shuffledListeCelluleSouche))).sum()
		if ishuffle in [0] :
			pilImC= Image.fromarray(labeledGridC.astype(np.int8))
			pilImC = pilImC.filter(ImageFilter.MaxFilter(3))
			labeledGridC = np.array(pilImC)
			exShuf = np.zeros(labeledGridC.shape)
			exShuf[np.isin(labeledGrid,shuffledListeCelluleSouche)]=2
			exShuf[labeledGridC ==1] = 1
			Image.fromarray(exShuf.astype(np.uint32)).save(imgSimuFileName[:imgSimuFileName.rfind('.')]+'_aShufflesGrid'+str(ishuffle)+'NbContact'+str(nbContactShuf)+'.tif')

		nbContactShuffle.append(nbContactShuf)

	return nbContactShuffle


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


##################### script
outputRep ='./dataExamples/ependymaP30/'

inputRep ='./dataExamples/ependymaP30/'

imgSegFileName = './dataExamples/ependymaP30/Fused_position9_P30_IAP_LAPP_HC.tif'

imgSimuFileName = './dataExamples/ependymaP30/allSimuOutput/unshuffled.tiff'

imgSyntheticRandFileName ='./dataExamples/ependymaP30/73_rand_oct2019.tif'

imgSyntheticUnClustFileName ='./dataExamples/ependymaP30/90_unclust_oct2019.tif'
imgSyntheticClustFileName ='./dataExamples/ependymaP30/623_Clustv2_oct2019.tif'
classFileName = './dataExamples/ependymaP30/P30_424_9_cells_features.csv'

fc = open(classFileName,'r')
lines = fc.readlines()
fc.close()

listeCelluleSouche=[]
for iline in xrange(1,len(lines)):
	lines[iline]=lines[iline].split(',')
	if lines[iline][2]=='stem':
		listeCelluleSouche.append(int(lines[iline][1]))

### img cell type on seg 
patternFile = Image.open(imgSegFileName)
imgPattern = np.array(patternFile).astype(np.int)
patternFile.close()
imC = label2Contours(imgPattern)
pilImC= Image.fromarray(imC.astype(np.int8))
pilImC = pilImC.filter(ImageFilter.MaxFilter(3))
imC = np.array(pilImC)
exSeg = np.zeros(imC.shape)
exSeg[np.isin(imgPattern,listeCelluleSouche)]=2
exSeg[imC ==1] = 1
Image.fromarray(exSeg.astype(np.uint32)).save(outputRep+'/SegCellType_stem.tif')

####

## distribution of the shuffled model : 
distri_nbContactShuffle_ObsSim = shuffleLabel(imgSimuFileName,listeCelluleSouche)
distri_nbContactShuffle_ObsSeg = shuffleLabel(imgSegFileName,listeCelluleSouche)
distri_nbContactShuffle_SyntheticRand = shuffleLabel(imgSyntheticRandFileName,listeCelluleSouche)
distri_nbContactShuffle_SyntheticUnClust = shuffleLabel(imgSyntheticUnClustFileName,listeCelluleSouche)
distri_nbContactShuffle_SyntheticClust = shuffleLabel(imgSyntheticClustFileName,listeCelluleSouche)

#distriGrid_nbContactShuffle_Sim = gridShuffling(imgSegFileName,listeCelluleSouche)#imgSimuFileName
distriGrid_nbContactShuffle_SyntheticRand = gridShuffling(imgSyntheticRandFileName,listeCelluleSouche)
#distriGrid_nbContactShuffle_SyntheticUnClust = gridShuffling(imgSyntheticUnClustFileName,listeCelluleSouche)
#distriGrid_nbContactShuffle_SyntheticClust = gridShuffling(imgSyntheticClustFileName,listeCelluleSouche)
distriGrid_nbContactShuffle_Sim = distriGrid_nbContactShuffle_SyntheticRand
distriGrid_nbContactShuffle_SyntheticRand = distriGrid_nbContactShuffle_SyntheticRand
distriGrid_nbContactShuffle_SyntheticUnClust = distriGrid_nbContactShuffle_SyntheticRand
distriGrid_nbContactShuffle_SyntheticClust = distriGrid_nbContactShuffle_SyntheticRand

nbContactShuffle = np.load(inputRep +'/nbContactShuffle_stemCells.npy')

nbContactSim_ObsSeg = extractObservedNumberOfContact(imgSegFileName,listeCelluleSouche) 
nbContactSim_ObsSim = extractObservedNumberOfContact(imgSimuFileName,listeCelluleSouche) 
nbContactSim_SyntheticRand = extractObservedNumberOfContact(imgSyntheticRandFileName,listeCelluleSouche)
nbContactSim_SyntheticUnClust = extractObservedNumberOfContact(imgSyntheticUnClustFileName,listeCelluleSouche)
nbContactSim_SyntheticClust = extractObservedNumberOfContact(imgSyntheticClustFileName,listeCelluleSouche)

import matplotlib as mpl
colorObsSim =(255/255.,88/255.,0/255.) #(159/255.,41/255.,94/255.)

fig = plt.figure(figsize=(4.5,15))
ax1 = fig.add_subplot(412)
valuesBins,binsLim, patches1 = plt.hist(nbContactShuffle,fc='none', lw=1.8, histtype='step',color=(96/255.,26/255.,162/255.),bins = np.array(range(np.min(nbContactShuffle),np.max(nbContactShuffle)+2))-0.5)
calculPvalue(nbContactSim_SyntheticClust,nbContactShuffle,valuesBins,binsLim, 'SyntheticClust')
valuesBins,binsLim, patches2 = plt.hist(distri_nbContactShuffle_SyntheticClust,fc='none', lw=1.5, histtype='step',color=(72/255.,112/255.,202/255.),bins = np.array(range(np.min(distri_nbContactShuffle_SyntheticClust),np.max(distri_nbContactShuffle_SyntheticClust)+2))-0.5)
calculPvalue(nbContactSim_SyntheticClust,distri_nbContactShuffle_SyntheticClust,valuesBins,binsLim, 'clustShufID')
valuesBins,binsLim, patches3 = plt.hist(distriGrid_nbContactShuffle_SyntheticClust,fc='none', lw=1., histtype='step',color=(52/255.,192/255.,237/255.),bins = np.array(range(np.min(distriGrid_nbContactShuffle_SyntheticClust),np.max(distriGrid_nbContactShuffle_SyntheticClust)+2))-0.5)
calculPvalue(nbContactSim_SyntheticClust,distriGrid_nbContactShuffle_SyntheticClust,valuesBins,binsLim, 'clustGrid')
lineObs = plt.vlines(nbContactSim_SyntheticClust,-1,100,color = colorObsSim,label = 'synthetic clustering')
#plt.vlines(64,-1,100,color='g')
plt.ylim(ymin=0)
plt.xlim(xmin=0,xmax = 160)
#plt.xlabel('number of clusters per image')
plt.ylabel('frequency')

lines = ax1.get_lines()

handles = [mpl.lines. Line2D((0,0),(0,1),color=c,lw = LW) for c,LW in [[(96/255.,26/255.,162/255.),1.8],[(72/255.,112/255.,202/255.),1.5 ], [(52/255.,192/255.,237/255.),1.]]]
first_legend = plt.legend(handles=handles,labels = ['our model','shuffle identity model','grid model'], bbox_to_anchor=(0.2, 1.))

# Add the legend manually to the current Axes.
ax = plt.gca().add_artist(first_legend)

plt.legend(handles=[lineObs], loc = 'best')

fig.add_subplot(411, sharex=ax1)

valuesBins,binsLim, patches = plt.hist(nbContactShuffle,fc='none', lw=1.8, histtype='step',color=(96/255.,26/255.,162/255.),bins = np.array(range(np.min(nbContactShuffle),np.max(nbContactShuffle)+2))-0.5)

#valuesBins,binsLim, patches = plt.hist(distri_nbContactShuffle_ObsSim,fc='none', lw=1.5, histtype='step',color=(72/255.,112/255.,202/255.),bins = np.array(range(np.min(distri_nbContactShuffle_ObsSim),np.max(distri_nbContactShuffle_ObsSim)+2))-0.5)
calculPvalue(nbContactSim_ObsSim,nbContactShuffle,valuesBins,binsLim, 'ObsSim')
valuesBins,binsLim, patches = plt.hist(distri_nbContactShuffle_ObsSim,fc='none', lw=1.5, histtype='step',color=(72/255.,112/255.,202/255.),bins = np.array(range(np.min(distri_nbContactShuffle_ObsSim),np.max(distri_nbContactShuffle_ObsSim)+2))-0.5)

valuesBins,binsLim, patches = plt.hist(distriGrid_nbContactShuffle_Sim,fc='none', lw=1., histtype='step',color=(52/255.,192/255.,237/255.),bins = np.array(range(np.min(distriGrid_nbContactShuffle_Sim),np.max(distriGrid_nbContactShuffle_Sim)+2))-0.5)
plt.vlines(nbContactSim_ObsSim,-1,100, linestyles = 'dashed',color = colorObsSim,label = 'reconstruction')
plt.vlines(nbContactSim_ObsSeg,-1,100, linestyles = 'dashed',color = (200/255.,50/255.,50/255.),label = 'segmentation')
#plt.vlines(64,-1,100,color='g')
plt.ylim(ymin=0)
plt.xlim(xmin=0,xmax = 160)
#plt.xlabel('number of clusters per image')
plt.ylabel('frequency')
plt.legend(loc = 'best')



fig.add_subplot(413, sharex=ax1)

valuesBins,binsLim, patches = plt.hist(nbContactShuffle,fc='none', lw=1.8, histtype='step',color=(96/255.,26/255.,162/255.),bins = np.array(range(np.min(nbContactShuffle),np.max(nbContactShuffle)+2))-0.5)
calculPvalue(nbContactSim_SyntheticRand,nbContactShuffle,valuesBins,binsLim, 'SyntheticRand')
valuesBins,binsLim, patches = plt.hist(distri_nbContactShuffle_SyntheticRand,fc='none', lw=1.5, histtype='step',color=(72/255.,112/255.,202/255.),bins = np.array(range(np.min(distri_nbContactShuffle_SyntheticRand),np.max(distri_nbContactShuffle_SyntheticRand)+2))-0.5)

valuesBins,binsLim, patches = plt.hist(distriGrid_nbContactShuffle_SyntheticRand,fc='none', lw=1., histtype='step',color=(52/255.,192/255.,237/255.),bins = np.array(range(np.min(distriGrid_nbContactShuffle_SyntheticRand),np.max(distriGrid_nbContactShuffle_SyntheticRand)+2))-0.5)
plt.vlines(nbContactSim_SyntheticRand,-1,100,color = colorObsSim,label = 'synthetic randomization')
#plt.vlines(64,-1,100,color='g')
plt.ylim(ymin=0)
plt.xlim(xmin=0,xmax = 160)
#plt.xlabel('number of clusters per image')
plt.ylabel('frequency')
plt.legend(loc = 'best')


fig.add_subplot(414, sharex=ax1)

valuesBins,binsLim, patches = plt.hist(nbContactShuffle,fc='none', lw=1.8, histtype='step',color=(96/255.,26/255.,162/255.),bins = np.array(range(np.min(nbContactShuffle),np.max(nbContactShuffle)+2))-0.5)
calculPvalue(nbContactSim_SyntheticUnClust,nbContactShuffle,valuesBins,binsLim, 'SyntheticUnClust')
valuesBins,binsLim, patches = plt.hist(distri_nbContactShuffle_SyntheticUnClust,fc='none', lw=1.5, histtype='step',color=(72/255.,112/255.,202/255.),bins = np.array(range(np.min(distri_nbContactShuffle_SyntheticUnClust),np.max(distri_nbContactShuffle_SyntheticUnClust)+2))-0.5)

valuesBins,binsLim, patches = plt.hist(distriGrid_nbContactShuffle_SyntheticUnClust,fc='none', lw=1., histtype='step',color=(52/255.,192/255.,237/255.),bins = np.array(range(np.min(distriGrid_nbContactShuffle_SyntheticUnClust),np.max(distriGrid_nbContactShuffle_SyntheticUnClust)+2))-0.5)
plt.vlines(nbContactSim_SyntheticUnClust,-1,100,color = colorObsSim,label = 'synthetic unclustering')
#plt.vlines(64,-1,100,color='g')
plt.ylim(ymin=0)
plt.xlim(xmin=0,xmax = 160)
plt.xlabel('number of contacts per image')
plt.ylabel('frequency')
plt.legend()

plt.savefig(outputRep+'/nbContactfig3_stem.pdf')
plt.clf()









