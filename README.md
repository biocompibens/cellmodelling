
-------

The *SET* folder contains the python codes of the SET (Synthesis of Epithelial Tissue) model developed in the Biocomp group of IBENS institute.

The *dataExamples* folder contains some data that are necessary to apply the provided examples of using the SET model.

The *analysis* folder contains the python codes used to analyse SET outputs generated for each application present in the related paper.

------

# SET 

The SET model builds an image tessellation that reconstruct a possible cell organization following a cell shape distribution extracted from a tissue segmentation.
Depending of the arguments, the model will : reconstruct the observed tissue or realize a random SET of the tissue with elliptic shaped cells, constrained the cell shape to be ellipses or to have a higher range of shape possibilities, ... 

## Configuration : 
run with python 2.7

use PIL, networkx, multiprocessing,tqdm,panda, cv2, packages

## Quick start : 
A basic example on P30 mice ependymal tissue
 1) download the following folder 
 2) open a console in the downloaded folder. 
  * to reconstruct the ependymal tissue (reconstruction by SET): 

```
python ./SET/model.py -l ./dataExamples/ependymaP30/Fused_position9_P30_segmentation.tif -o ./dataExamples/ependymaP30/reconstruction 
```

  * to generate 1000 randomly shuffled ependymal tissues (random SET) :


```
python ./SET/model.py -l ./dataExamples/ependymaP30/Fused_position9_P30_segmentation.tif -o ./dataExamples/ependymaP30/allSimuOutput -s 1000
```

Note that the random SET are generated one after the others and it can take long time. To accelerate the calcul, 
1) use the -n option and specify a number of CPU to use multiprocessing 
2) use a cluster to realize in parallele several unit shuffle simulation  (-s 1 instead of -s 1000) 
3) anticipate the use of a mean measurement to analyse your observation of interest : the distribution a mean value can be approximate with only one occurence of the distribution with a gaussian <img src="http://latex.codecogs.com/svg.latex?(\mu,\frac{\sigma}{\sqrt{n}})" border="0"/>, n the number of measures in the distribution.

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
python ./SET/model.py  -l ./dataExamples/ependymaE18/E18\_24\_series51channel2WFGM22LABEL.tif -o ./dataExamples/ependymaE18 -p 5 -wp ./dataExamples/ependymaE18/centrioleDetection_s51.npy
```

## Reference: 



