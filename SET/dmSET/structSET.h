typedef struct setImage setImage;
struct setImage{
	
	int * nbPxS;
	int * nbPxC;

	int nbLabels;
	
	//int nbPxS = 0;
	//int nbPxC = 0;

	unsigned long height;
	unsigned long width;

	long* segmentation ; 
	short* ske ; 
	long* contours ;

	double* map ;
	unsigned long* labels ;

	int** pxC ;
	int** pxS ;

};
