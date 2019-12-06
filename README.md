
# SET 

## Description

The SET method fit a flexible distance function on each cell of a segmented image and use them to generate a cell tesselation. The resulting tesselation is either a close reconstruction of the actual image, either an image of cell tesselation that is made of the same cells organized randomly. 

The SET method can be used to generate a null distribution in order to statistically test for the significance of an observed pattern.

## Repository content

The *SET* folder contains the python codes to generate reconstruction and random SET (Synthesis of Epithelial Tissue) from an image.

The *dataExamples* folder contains exemple images to be processed by SET.

The *analysis* folder contains the python codes used in the paper to analyse SET outputs and generate statistics.


## Configuration : 
python 2.7, matplotlib 3.0.0, numpy 1.16.2, scipy 1.2., PIL 4.2.1, networkx 2.2, multiprocessing 0.70a1, tqdm 4.20.0,panda 0.23.4, cv2 2.4.9.11 packages


## Quick start exemple : 
Example: P30 mice ependymal tissue
* to reconstruct the cell tesselation from an actual image (reconstruction by SET): 

```
python ./SET/model.py -l ./dataExamples/ependymaP30/Fused_position9_P30_segmentation.tif -o ./dataExamples/ependymaP30/reconstruction 
```

* to generate 1000 randomly shuffled cell tesselations from the cells of the same image (random SET):


```
python ./SET/model.py -l ./dataExamples/ependymaP30/Fused_position9_P30_segmentation.tif -o ./dataExamples/ependymaP30/allSimuOutput -s 1000
```

Note that the 1000 random SET are generated sequencially so that it can take a long time. To speed up the process you may: 
1. use the -n option to specify a max number of CPUs to use simultaneously such that the generation of each SET will be parallelized on n CPUs. Still the SET will be generated sequencially. Note that the creation of the reconstruction by SET also support the -n option. 
2. use a computing cluster to process in parallel random SETs. In this case a 1000 jobs with the option "-s 1" can be considered. Note that each job can still be also parallelized using the -n option. Note also that each job can be set to generate a subset of random SETs. For instance, 100 jobs with the option "-s 10" can be used to generate 1000 random SETs.  
3. consider the sample mean distribution of a cell to cell relationship as feature of interest as it can be approximated by a Gaussian distribution from a single (or a few) random SET with parameters <img src="http://latex.codecogs.com/svg.latex?(\mu,\frac{\sigma}{\sqrt{n}})" border="0"/> following the Central Limit Theorem. n the number of measures in the sample.

* to run the dedicated analysis that compute stem cell coupling counts from the reconstruction by SET and the distribution of random SET:

```
python ./analysis/p30/p30pos9_contactStemCells.py
```

## model.py arguments details : 

* -l : image of cell label (image file path). All pixel of this image must take as a value an integer corresponding to a unique cell.
* -s : generate as many random SET as specified by the specified number (integer).
* -o : name of the output folder (folder path)
* -p : number of parameters used to construct the distance function. Can be 5 (Mahalanobis distance model cells as ellipses), 6 (MAT distance model cells as superellipses), 7 (Asymetric distance model cells as ovoidal asymetric shapes), 8 (AMAT distance includes all parameters). Default is 5.
* -i : input raw image (greylevel or RGB), activates morphing of the intracellular contents (.tif, png or ...)
* -n : maximum number of simulteneously running CPU for parallel processing (integer)
* -d : path movie directory name (folder path). Option to generate an image for each Lloyd iteration.
* -sp : subpopulation shuffling. Must specify 
	 1) the name (path) of the file containing cell class (.npy , .csv or .tsv) 
	 2) the class names that should be shuffle (can be more than 1)
* -wp : option to morph only a position of an intracellular component. Need the path of the file containing the position of an organel per label (.npy, .csv or.tsv)
* -r : reprocess a SET. Need the path of the csv generated by the SET (.tsv)
* -it : max number of iterations for the Lloyd algorithm . Default is 80 which is valid for most application.

## Output : 
.tiff image file including the labels of the SET tesselation, these label values match the label values of the image input (cell i corresponds to synthtetic cell i)

.tsv containing parameters per cells extracted from the segmentation, the shuffled positions and then the final values of the parameters

-wp option generates a .tsv containing the computed position of the intracellular marker per cell in the synthtetic cells

-i option generates a .png where each original cell texture is moved into its new location in the SET.

-d option generates 2 .png files per Lloyd iteration : 
	1) the image of SET where cell labels indicates where each original cell is
	2) the image of SET where each cell content is filled with the original cell texture



## Other examples : 

* Analysis of intercellular components relationships : the centriole orientation of a E18 ependymal tissue

```
python ./SET/model.py  -l ./dataExamples/ependymaE18/E18_24_series51channel2WFGM22LABEL.tif -o ./dataExamples/ependymaE18 -p 5 -wp ./dataExamples/ependymaE18/centrioleDetection_s51.npy
```

## Reference: 



