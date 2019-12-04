## code to analyze centriole orientation per neig rank
### this code run with multiprocessing  : change the nbCPU variable 
## this code need to have generate random set and the reconstruction by set with : 
## python ./SET/model.py  -l ./dataExamples/ependymaE18/E18_24_series51channel2WFGM22LABEL.tif -o ./dataExamples/ependymaE18/allSimuOutput -p 5 -wp ./dataExamples/ependymaE18/centrioleDetection_s51.npy -s 1000

import multiprocessing
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import numpy as np
from PIL import Image
import os
import glob
import skimage.future
import scipy.ndimage
import sys
import scipy.stats as st


shufflePositionFiles = glob.glob('./dataExamples/ependymaE18/allSimuOutput/[0-9]*_centriolfinalposition.txt')
unshuffleFile = glob.glob('./dataExamples/ependymaE18/allSimuOutput/unshuffled_centriolfinalposition.txt')
shuffleLabelImg = glob.glob('./dataExamples/ependymaE18/allSimuOutput/[0-9]*.tiff')
unshuffleLabelImg = glob.glob('./dataExamples/ependymaE18/allSimuOutput/unshuffled.tiff')

def label2Contours(ImgL1):

	region_borders = scipy.ndimage.morphological_gradient(ImgL1,
                                            footprint=[[0, 1, 0],
                                                       [1, 1, 1],
                                                       [0, 1, 0]])
	yxB = np.argwhere(region_borders!=0)
	region_borders[yxB[:,0],yxB[:,1]]=1
	return region_borders 

def analyseOneOfTheShuffleImage(indx,imageFileName,data,return_dict) :
	if data == []:
		return_dict[indx] = [np.nan,np.nan]
	else : 

		fimg = Image.open(imageFileName)
		img = np.array(fimg)
		fimg.close()

		labels = np.unique(img)
		if 0 and len(labels) !=1091 :
			return_dict[indx] = [np.nan,np.nan]
		else :

			imC = label2Contours(img)

			ragShuff = skimage.future.graph.rag_boundary(img, imC)
			keysRag= np.array(ragShuff.edges.keys())

			boundCells = np.unique(np.concatenate([np.unique(img[:,0]),np.unique(img[:,-1]),np.unique(img[-1,:]),np.unique(img[0,:])]))
			lxy = []
			for line in xrange(len(data)):
				l = data[line].split('\t')
				if 'there is an error' in l :
					lxy.append([float(l[0]),np.nan,np.nan])
				else :
					try:
						lxy.append(map(float,l))
					except :
						lxy.append([float(l[0]),np.nan,np.nan])
			d = np.array(lxy)[:,1:3]
			coordMean = []
			for eachL in xrange(len(d)):
				if np.array(lxy)[eachL,0] in labels: 
					yx = np.argwhere(img == np.array(lxy)[eachL,0])
					coordMean.append(np.mean(yx, axis=0))
				else : 
					coordMean.append([np.nan,np.nan])
			coordMean=np.array(coordMean)

			distanceBetweenCells  = np.zeros((len(labels),len(labels)),dtype=int)
			missingLabel = []
			for iCell in xrange(len(labels)):
				if not(labels[iCell] in boundCells):
					neigNew = [labels[iCell]]
					neigOld = [labels[iCell]]
					loop=1
					while len(neigNew) !=0:
						neig = []
						for nold in neigNew:
							try :
								neig1 = keysRag[np.concatenate(np.argwhere(keysRag[:,0]==nold)),1]
							except :
								neig1 = []
							try :
								neig2 = keysRag[np.concatenate(np.argwhere(keysRag[:,1]==nold)),0]
							except :
								neig2 = []
							neig = neig + list(neig1) + list(neig2)
							if len(neig) == 0 :
								missingLabel.append(nold)
								print "cell "+str(nold)+" not in RAG"+'\n'
								continue
						neigNew = list(set(np.unique(neig)) - set(neigOld) - set(boundCells))
						idNeig = np.squeeze(np.argwhere(np.isin(labels, neigNew)))
						distanceBetweenCells[idNeig,iCell] = loop
						neigOld = list(neigNew)+neigOld
						loop +=1

			loopCouple = [np.unique(np.sort(np.argwhere(distanceBetweenCells==i),axis=1),axis = 1) for i in xrange(1,np.max(distanceBetweenCells))]

			anglePerLoop = []
			for iLoop in xrange(len(loopCouple)):
				anglePerLoop.append([])
				for eachC in xrange(len(loopCouple[iLoop])):
					idxC1 = loopCouple[iLoop][eachC,0]
					idxC2 = loopCouple[iLoop][eachC,1]
					iC1 = np.squeeze(np.argwhere(np.array(lxy)[:,0] == labels[idxC1]))
					iC2 = np.squeeze(np.argwhere(np.array(lxy)[:,0] == labels[idxC2]))
					if np.isnan(d[iC1,0]) or np.isnan(d[iC2,0]) :
						continue
					vNeig = [np.squeeze(d[iC1,0]-coordMean[iC1,0]),np.squeeze(d[iC1,1]-coordMean[iC1,1])]
					vCell = [np.squeeze(d[iC2,0]-coordMean[iC2,0]),np.squeeze(d[iC2,1]-coordMean[iC2,1])]

					cosAlpha= np.clip((vNeig[0]*vCell[0]+vNeig[1]*vCell[1])/(np.sqrt(np.power(vNeig[0],2)+np.power(vNeig[1],2))*np.sqrt(np.power(vCell[0],2)+np.power(vCell[1],2))), -1.0, 1.0)
					alpha =np.arccos(cosAlpha)
					anglePerLoop[iLoop].append(alpha)
			np.save(imageFileName[:imageFileName.rfind('.')]+'NeigLoop_withoutBound',anglePerLoop)


			return_dict[indx] = [anglePerLoop]

reso = [0.0901435,0.0901435]

#shuf analysis
nbCPU =  1

manager = multiprocessing.Manager()
return_dict = manager.dict()
iImg = 0
while iImg <len(shuffleLabelImg) :
    jobs = []
    for i in range(nbCPU):
        if iImg >=len(shuffleLabelImg):
            break

        idShuff = shuffleLabelImg[iImg][shuffleLabelImg[iImg].rfind('/')+1:shuffleLabelImg[iImg].rfind('.')]
        for eachPosFile in xrange(len(shufflePositionFiles)) :
			if idShuff in shufflePositionFiles[eachPosFile]:
				f = open(shufflePositionFiles[eachPosFile])
				data = f.readlines()
				f.close()
				break
        if not(idShuff in shufflePositionFiles[eachPosFile]):
			print 'no centriole position file for : ' + idShuff
			data = []
        p = multiprocessing.Process(target=analyseOneOfTheShuffleImage, args=(iImg,shuffleLabelImg[iImg],data,return_dict))
        iImg +=1
        print iImg
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()


try : 
	anglePerLoopShuff = []
	for i in xrange(len(return_dict.values())):
		if type(return_dict[i][0]) == list:
			anglePerLoopShuff.append(return_dict[i][0])
except :
	anglePerLoopShuff = []
	for iImg in xrange(len(shuffleLabelImg)):
		imageFileName = shuffleLabelImg[iImg]
		anglePerLoopShuff.append(np.load(imageFileName[:imageFileName.rfind('.')]+'NeigLoop_withoutBound.npy'))


##unshuf analysis
fimg = Image.open(unshuffleLabelImg[0])
img = np.array(fimg)
fimg.close()
labels = np.unique(img)

f = open(unshuffleFile[0])
data = f.readlines()
f.close()

imC = label2Contours(img)

ragShuff = skimage.future.graph.rag_boundary(img, imC)
keysRag= np.array(ragShuff.edges.keys())

boundCells = np.unique(np.concatenate([np.unique(img[:,0]),np.unique(img[:,-1]),np.unique(img[-1,:]),np.unique(img[0,:])]))

lxy = []
for line in xrange(len(data)):
	l = data[line].split('\t')
	if 'there is an error' in l :
		lxy.append([float(l[0]),np.nan,np.nan])
	else :
		try:
			lxy.append(map(float,l))
		except :
			lxy.append([float(l[0]),np.nan,np.nan])
d = np.array(lxy)[:,1:3]


coordMean = []
for eachL in xrange(len(d)):
	if np.array(lxy)[eachL,0] in labels: 
		yx = np.argwhere(img == np.array(lxy)[eachL,0])
		coordMean.append(np.mean(yx, axis=0))
		irand = np.random.randint(len(yx))
	else : 
		coordMean.append([np.nan,np.nan])
coordMean=np.array(coordMean)

distanceBetweenCells  = np.zeros((len(labels),len(labels)),dtype=int)
missingLabel = []
for iCell in xrange(len(labels)):
	if not(labels[iCell] in boundCells):
		neigNew = [labels[iCell]]
		neigOld = [labels[iCell]]
		loop=1
		while len(neigNew) !=0:
			neig = []
			for nold in neigNew:
				try :
					neig1 = keysRag[np.concatenate(np.argwhere(keysRag[:,0]==nold)),1]
				except :
					neig1 = []
				try :
					neig2 = keysRag[np.concatenate(np.argwhere(keysRag[:,1]==nold)),0]
				except :
					neig2 = []
				neig = neig + list(neig1) + list(neig2)
				if len(neig) == 0 :
					missingLabel.append(nold)
					print "cell "+str(nold)+" not in RAG"+'\n'
					continue
			neigNew = list(set(np.unique(neig)) - set(neigOld) - set(boundCells)) 
			idNeig = np.squeeze(np.argwhere(np.isin(labels, neigNew)))
			distanceBetweenCells[idNeig,iCell] = loop

			neigOld = list(neigNew)+neigOld
			loop +=1

loopCouple = [np.unique(np.sort(np.argwhere(distanceBetweenCells==i),axis=1),axis = 1) for i in xrange(1,np.max(distanceBetweenCells))]

anglePerLoop = []
for iLoop in xrange(len(loopCouple)):
	anglePerLoop.append([])
	eachC = len(loopCouple[iLoop])-1
	print iLoop
	while eachC>=0 :	
		idxC1 = loopCouple[iLoop][eachC,0]
		idxC2 = loopCouple[iLoop][eachC,1]
		iC1 = np.squeeze(np.argwhere(np.array(lxy)[:,0] == labels[idxC1]))
		iC2 = np.squeeze(np.argwhere(np.array(lxy)[:,0] == labels[idxC2]))
		if np.isnan(d[iC1,0]) or np.isnan(d[iC2,0]) :
			loopCouple[iLoop] = np.concatenate([loopCouple[iLoop][:eachC ],loopCouple[iLoop][eachC+1 :]])
			eachC-=1
			continue
		vNeig = [np.squeeze(d[iC1,0]-coordMean[iC1,0]),np.squeeze(d[iC1,1]-coordMean[iC1,1])]
		vCell = [np.squeeze(d[iC2,0]-coordMean[iC2,0]),np.squeeze(d[iC2,1]-coordMean[iC2,1])]
		cosAlpha= np.clip((vNeig[0]*vCell[0]+vNeig[1]*vCell[1])/(np.sqrt(np.power(vNeig[0],2)+np.power(vNeig[1],2))*np.sqrt(np.power(vCell[0],2)+np.power(vCell[1],2))), -1.0, 1.0)
		alpha =np.arccos(cosAlpha)
		anglePerLoop[iLoop].append(alpha)
		eachC -=1


np.save('./dataExamples/ependymaE18/anglePerLoop_withoutBound',anglePerLoop)
np.save('./dataExamples/ependymaE18/anglePerLoopShuff_withoutBound',anglePerLoopShuff)

#anglePerLoop = np.load('/data/biocomp/laruelle/E18centrioles_V2/anglePerLoop.npy')

### graphs
import scipy.stats as st
distribution = st.norm
manglePerLoop = [np.nanmean(a) for a in anglePerLoop]
maxNeigRank = np.min(map(len,anglePerLoopShuff))
xLoop= range(1, maxNeigRank)
manglePerLoopShuff = np.array([[np.nanmean(s[l]) if (len(s)>l and len(s[l])!=0) else np.nan  for s in anglePerLoopShuff ] for l in xrange(maxNeigRank) ])

manglePerLoopShuff = np.array(manglePerLoopShuff)
np.save('./dataExamples/ependymaE18/manglePerLoopShuff',manglePerLoopShuff)

plt.clf()
fig = plt.figure(figsize=(5,60))
paramPerLoop = []
for eachLo in xrange(len(xLoop)) :
	ax = fig.add_subplot(len(xLoop),1,eachLo+1)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	shufDistriThisLoop = np.degrees(manglePerLoopShuff[xLoop[eachLo]-1,:])
	shufDistriThisLoop = shufDistriThisLoop[~np.isnan(shufDistriThisLoop)]
	#plt.hist(shufDistriThisLoop, color = col[xLoop[eachLo]-1], alpha = 0.6) ##uncomment to show the distribution 
	obsValue = np.degrees(manglePerLoop[xLoop[eachLo]-1])
	plt.vlines(obsValue, 0,1, color = (81/255.,211/255.,52/255.), label = str(xLoop[xLoop[eachLo]-1]))

	lenShuff = float(len(shufDistriThisLoop))
	if obsValue>np.mean(shufDistriThisLoop) :
		proba = len(np.argwhere(np.array(shufDistriThisLoop)>=obsValue))/lenShuff
	else :
		proba = len(np.argwhere(np.array(shufDistriThisLoop)<=obsValue))/lenShuff
	if proba == 0 :
		proba = 1./(lenShuff+1)

	plt.text(92, 1, 'rank '+ str(xLoop[eachLo]), color = (53/255.,149/255.,247/255.))


	params = distribution.fit(shufDistriThisLoop)
	fitted = distribution(*params)
	paramPerLoop.append(params)
	print fitted.mean(),fitted.std(), fitted.cdf(obsValue)
	x = np.linspace(0,180,num =1000)
	yFit = fitted.pdf(x)
	plt.plot(x,yFit ,  lw=2, color=(53/255.,149/255.,247/255.), alpha = 0.7) #, label= 'fitted gaussian'

	plt.xlim(xmin=80,xmax=95)#(xmin=0,xmax=180)
	plt.ylim(ymax=2)
	xt = range(80, 95+1, 5)
	plt.xticks(xt, [str(i)+'$^\circ$' for i in xt] )
	plt.ylabel("density")
plt.xlabel('mean absolute angle (degree)')
plt.savefig('./dataExamples/ependymaE18/meanPerNeigRank_subPlots.pdf')


