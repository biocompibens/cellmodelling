
# SET 

The SET method fit a flexible distance function on each cell of an image and use them to generate a cell tesselation. The resulting tesselation is either a close reconstruction of the actual image, either an image of cell tesselation that is made of the same cells organized randomly. 

The SET method can be used to generate a null distribution in order to statistically test for the significance of an observed pattern.

# Content description

The *SET* folder contains the python codes to generate reconstruction and random SET (Synthesis of Epithelial Tissue) from an image.

The *dataExamples* folder contains exemple images to be processed by SET.

The *analysis* folder contains the python codes used in the paper to analyse SET outputs and generate statistics.


## Configuration : 
python 2.7, PIL, networkx, multiprocessing,tqdm,panda, cv2, packages

## Quick start : 
A basic example on P30 mice ependymal tissue
  * to only reconstruct the ependymal tissue (reconstruction by SET): 

```
python ./SET/model.py -l ./dataExamples/ependymaP30/Fused_position9_P30_segmentation.tif -o ./dataExamples/ependymaP30/reconstruction 
```

  * to generate 1000 randomly shuffled ependymal tissues (random SET) on top of the reconstruction:


```
python ./SET/model.py -l ./dataExamples/ependymaP30/Fused_position9_P30_segmentation.tif -o ./dataExamples/ependymaP30/allSimuOutput -s 1000
```

Note that the random SET are generated sequencially so it can take long time. To accelerate the process you can: 
1) use the -n option to specify a number of CPUs such that the process will be parallelized 
2) use a computing cluster to process in parallel random SET  (-s 1 instead of -s 1000 but on 1000 jobs) 
3) consider a mean measurement over items to analyse your observation of interest : the distribution of a mean value can be approximated with only one occurrence of the distribution with a Gaussian <img src="http://latex.codecogs.com/svg.latex?(\mu,\frac{\sigma}{\sqrt{n}})" border="0"/>, n the number of measures in the distribution.

  * comput statistics :

```
python ./analysis/p30/p30pos9_contactStemCells.py
```


## Arguments details : 

* -l : label image path of the studied tissue (tif, png or ...)
* -s : option to shuffle randomly cell positions. Must be followed by the number of random SET to generate. 
* -o : name folder output (path)
* -p : number of parameters used to construct the shape distance function. Default is 5 (ellipses). Can be 5 (mahalanobis distance function), 6 (Minkowski metrics), 7 (asymmetric elliptic shape), 8 (all parameters).
* -i : input raw image (RGB or not), when present : morphing of the intracellular contents is activate. (.tif, png or ...)
* -n : number of jobs for parallel processing
* -d : path movie directory name. Option to generate each Lloyd iteration image.
* -sp : subpopulation shuffling option. Must precise 
	 1) the name (path) of the file containing cell classification (.npy) 
	 2) then the class names that you want to shuffle (can be more than 1)
* -wp : option to morph only a position of an intracellular component. Need the path of the file containing the position per label (.npy)
* -r : redo a simulation. Need the path of the csv generate during the first simulation (.csv)
* -in : max number of iterations realized by the Lloyd algorithm to consider the convergence. By default, it is 80.

## Output : 
.tiff image file containing the last tessellation, labeled correspondingly to the labeled image input.

.csv containing parameters per cells extracted from the segmentation, then the final values of the parameters

-wp option generates a .txt containing the new position of the intracellular marker per cell

-i option generates a .png containing the intracellular morphing image

-d option generates 2 .png files : 
	1) the labels 
	2) the corresponding shapes draw over border image



## Other examples : 

* Analysis of intercellular components relation : the centriole orientation of a E18 ependymal tissue

```
python ./SET/model.py  -l ./dataExamples/ependymaE18/E18_24_series51channel2WFGM22LABEL.tif -o ./dataExamples/ependymaE18 -p 5 -wp ./dataExamples/ependymaE18/centrioleDetection_s51.npy
```

## Reference: 



