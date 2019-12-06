
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import itertools
import numpy as np
import argparse
import shutil
import os
import cv2
import sys

from tqdm import tqdm
from scipy.spatial.distance import cdist
from synthesis.warping import warp
from PIL import Image
from skimage.measure import label
from scipy.ndimage import morphological_gradient

from synthesis.profiling import Profiles
from synthesis.modeling import Lloyd
from synthesis.wrapingPix import points_synthesis




##########
#import warnings

#warnings.filterwarnings('error')
##########

def readOutputSimFile(fileName):
	f = open(fileName)
	data = f.readlines()
	f.close()
	init = []
	shuf = []
	out = []
	flag = 0
	for line in range(len(data)):
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
	return init[1:,0].astype(int),init[1:,1].astype(float),init[1:,2].astype(float),init[1:,3].astype(float),init[1:,4].astype(float),init[1:,5].astype(float),init[1:,6].astype(float),init[1:,7].astype(float),init[1:,8].astype(float),shuf[1:,0].astype(int),shuf[1:,1].astype(float),shuf[1:,2].astype(float),shuf[1:,3].astype(float),out[1:,0].astype(int),out[1:,1].astype(float),out[1:,2].astype(float),out[1:,3].astype(float)

def readClassOfSubpop(filename)
	extension = filename[filename.rfind('.'):]	
	if 	extension == '.npy':
		cellClass = np.array(np.load(fileName))
	elif extension == '.csv':
		f = open(filename,'r')
		data = f.readlines()
		f.close()
		for iLine in xrange(len(data)):
			data[iLine] = data[iLine].split(',')
			data[iLine] = [int(data[iLine][0]),data[iLine][1][:-1]]
		cellClass = np.array(data)
	elif extension == '.tsv':
		f = open(filename,'r')
		data = f.readlines()
		f.close()
		for iLine in xrange(len(data)):
			data[iLine] = data[iLine].split('\t')
			data[iLine] = [int(data[iLine][0]),data[iLine][1][:-1]]
		cellClass = np.array(data)
	return cellClass

			cellClass = np.load(shufpopInfo[0])

if __name__ == "__main__":

	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, usage="")
	parser.add_argument("-i", "--cellimagefile", type=str, help="Cell image file\n\n")
	parser.add_argument("-l", "--celllabelfile", type=str, help="Cell label file\n\n")
	parser.add_argument("-o", "--outPath", type=str, help="OutPath name\n\n")
	parser.add_argument("-n", "--njobs", type=int, default=1, help="Number of parallel jobs\n\n")
	parser.add_argument("-s", "--shuffle", help="number of shuffled images - shuffle cells\n\n")
	parser.add_argument("-d", "--dirmovie", help="Save process movie in the specified directory\n\n")
	parser.add_argument("-sp", "--spop", nargs='*', help="shuffle one population- class file- class to shuffle\n\n")
	parser.add_argument("-p", "--parameters", type=str, default='',help="number of used parameters (5(ellipse),6(p-only),7(a-only),8(all))\n\n")
	parser.add_argument("-wp", "--warpingPoint", type=str, help="point file\n\n")
	parser.add_argument("-r", "--redo",  type=str,help="redo a simulation from parameters\n\n")
	parser.add_argument("-it", "--iterationNumber",  type=int,help="number of iterations\n\n")


	if len(sys.argv[1:])==0:
		parser.print_help()
		parser.exit()

	options = parser.parse_args()

	imagefile = options.cellimagefile 
	labelfile = options.celllabelfile
	outPath = options.outPath
	n_jobs = options.njobs
	shuffle = options.shuffle
	dirmovie = options.dirmovie
	shufpopInfo = options.spop
	_paramShape = options.parameters
	pointFileName = options.warpingPoint
	redoFile = options.redo
	_interNumber = option.iterationNumber

	if _interNumber== None:
		_interNumber = 80

	if _paramShape == None or _paramShape == '5' or _paramShape == 'ellipse':
		_paramShape = ''

	if imagefile == None:
		imagefile = ''

	im_labels_orig = np.array(Image.open(labelfile)).astype(int)
	if imagefile == '':
		im_values_orig = np.zeros((im_labels_orig.shape[0],im_labels_orig.shape[1],3))
	else :
		im_values_orig = np.array(Image.open(imagefile))
		if len(im_values_orig.shape)!=3 :
			im_values_orig = np.concatenate(im_values_orig,np.zeros((im_labels_orig.shape[0],im_labels_orig.shape[1],3)),axis=2)


	profiles = Profiles.from_image(im_values_orig, im_labels_orig, include_border=True , n_jobs=n_jobs,_p = _paramShape) #, n_jobs=70)#
	
	height, width = im_labels_orig.shape[:2]
	
	if not (dirmovie is None):
		shutil.rmtree(dirmovie, ignore_errors=True)		
		os.makedirs(dirmovie)
	

	id =  np.array(profiles.df.id, dtype=int)
	rid = np.zeros(np.max(id) + 1, dtype=int)
	rid[id] = np.arange(id.shape[0]) #reverse ids

	border =  np.array(profiles.df.border, dtype=bool)
	
	params_orig = np.array(profiles.df.loc[:,['x', 'y', 'theta', 'l1', 'l2', 'a1', 'a2', 'p']], dtype='float64')
	params = np.array(profiles.df.loc[:,['x', 'y', 'theta']], dtype='float64')
	if shuffle : 
		shuffle = int(shuffle)

	if  not(os.path.exists(outPath)): 
		os.makedirs(outPath)

	for iSimu in range(max([1,shuffle])) :
		if shuffle:
			outfile = outPath +("/%04d"%iSimu)+".tiff"
			fOutfile = outPath +("/%04d"%iSimu)+'sLCSTable.tsv'
		else : 
			outfile = outPath +"/unshuffled.tiff"
			fOutfile = outPath +"/unshuffled"+'uLCSTable.tsv'
		print([outfile,fOutfile])

		if redoFile :
			id,_,_,_,l1,l2,a1,a2,p,_,Cx,Cy,theta,_,_,_,_ = readOutputSimFile(redoFile)
			C = [Cx,Cy]
			params[:,(0,1)] = np.array(C).T
			fOutfile = fOutfile[:-4]+'_re.tsv'

		C = params[:,(0,1)]
		angles = params[:,2]

		I = np.array(list(itertools.product(list(range(height)),list(range(width)))))


		labels = np.argmin(cdist(I, C), axis=1) #Voronoi, range 0..n
	
		if shufpopInfo!= None :
			labelIndx = np.unique(im_labels_orig)
			cellClass = readClassOfSubpop(shufpopInfo[0])
			gobletSimuLabels=[]
			for popToshuffle in range(1,len(shufpopInfo)):
				gobletSimuLabels.append( np.squeeze(labelIndx[np.argwhere(cellClass==str(shufpopInfo[popToshuffle]))]) )
			gobletSimuLabels = np.concatenate(gobletSimuLabels)
			boolCelluleSouche = [  ilabel in gobletSimuLabels for ilabel in labelIndx]
			for eachC in range(len(C)): #range(1,len(C)) --> pourquoi 1 ?? je le faisais pas pour ependyme 
				if boolCelluleSouche[eachC]==1 and not(border[eachC]) :
					C[eachC,:] = [np.random.randint(width+1),np.random.randint(height+1)]
			params[:,(0,1)] = C


		# file to extract shape parameters information

		simFeaturesFile = open(fOutfile,'w')
		simFeaturesFile.write('id\tCx\tCy\ttheta\tl1\tl2\ta1\ta2\tp\n')
		for ilabel in range(len(id)):
			simFeaturesFile.write(str(id[ilabel]) + '\t' +str(C[ilabel][0]) +'\t' +str(C[ilabel][1])+'\t'+ str(params_orig[ilabel][2]) +'\t' +str(params_orig[ilabel][3])+'\t'+str(params_orig[ilabel][4]) +'\t' +str(params_orig[ilabel][5])+'\t'+str(params_orig[ilabel][6])+'\t'+str(params_orig[ilabel][7])+'\n')
		simFeaturesFile.close()


		labels = np.argmin(cdist(I, C), axis=1)
		#SHUFFLE CELLS: random sample within non border cells (+random angles)
		if shuffle:
			nbidx = np.where(~np.array(border))[0]
			Inotborder = I[~np.array(border)[labels]]
			C[nbidx] = Inotborder[np.random.choice(np.arange(Inotborder.shape[0]), nbidx.shape[0], replace=False)]
			angles[nbidx] = np.random.uniform(-np.pi, np.pi, nbidx.shape[0])
			# store back into params
			params[:,(0,1)] = C
			params[:,2] = angles
		
			labels[~np.array(border)[labels]] = -1
			labels_reshape = labels.reshape((height, width))
			nonborder_idx = np.where(labels_reshape == -1)

			labels_reshape[nonborder_idx] = nbidx[np.argmin(cdist(np.array(nonborder_idx).T, C[nbidx]), axis=1)]
			labels = labels_reshape.reshape(width*height)
			
		simFeaturesFile = open(fOutfile,'a')
		simFeaturesFile.write('id_s\tCx_s\tCy_s\ttheta_s\n')
		for ilabel in range(len(id)): #pas sure que ce soit dans le bon ordre
			simFeaturesFile.write(str(id[ilabel]) + '\t' +str(C[ilabel][0]) +'\t' +str(C[ilabel][1])+'\t'+ str(params[ilabel][2]) +'\n')
		simFeaturesFile.close()


		im_labels_orig_sid = rid[im_labels_orig] #converting original labels to 0..n
	
		paramsOut, labels = Lloyd(I, labels, params, params_orig, border, im_labels_orig_sid, im_values_orig, n_jobs=n_jobs, max_iter=_interNumber, dir=dirmovie, pShape =_paramShape)

		if pointFileName :

			imD,xyc =points_synthesis(pointFileName, labels, paramsOut[:,0:2], im_values_orig,  im_labels_orig_sid, paramsOut[:,2], params_orig[:,2], params_orig[:,3:5], id, outfile, n_jobs=n_jobs, display=True)
			simCentriolFile = open(outfile[:outfile.rfind('.')]+'_centriolfinalposition.tsv','w')
			for ilabel in range(len(id)): 
				try :
					simCentriolFile.write(str(id[ilabel]) + '\t' +str(xyc[ilabel][0]) +'\t' +str(xyc[ilabel][1])+'\n')
				except :
					simCentriolFile.write(str(id[ilabel])+ '\t' + 'there is an error')
			simCentriolFile.close()


		labels = id[labels] # converting back to the original ids

		Image.fromarray(labels.astype(np.uint16).reshape((height,width))).save(outfile)

		# add simulation shape information to the file
		simFeaturesFile = open(fOutfile,'a')
		simFeaturesFile.write('id_f\tCx_f\tCy_f\ttheta_f\n')
		for ilabel in range(len(id)): 
			simFeaturesFile.write(str(id[ilabel]) + '\t' +str(paramsOut[ilabel,0]) +'\t' +str(paramsOut[ilabel,1])+'\t'+ str(paramsOut[ilabel,2]) +'\n')
		simFeaturesFile.close()

