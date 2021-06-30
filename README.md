## Pre-requirements 
1. Environments and required packages
    + python>=3.5
    + matlab
    + numpy
    + scipy 
    + scikit-learn (joblib supported) 
    + tqdm
2. Brain imaging data
    + We here use the imaging data published by Pereira et al., in their [Nature Communications](https://www.nature.com/articles/s41467-018-03068-4) paper. Plz refer the paper for details of the imaging data. 
    + The data is publicly available at https://osf.io/crwz7/. We specifically use the preprocessed functional data, for example "data_384sentences.mat" of each subject.
    + The brain activation patterns need to be extracted from the .mat file into numpy format, with shape "stimulus number * voxel number ". 
       + They should be put in one directory with mutiple subdirectories named by dubject id.
       + In our case, this directory is named "voxel", and subject IDs include M02, M04, M07 et al

3. Word or sentence representation
    + The text representations need to be saved in numpy format, with shape "stimulus number * embedding dimension". 
4. Informativeness score of voxels (needed by decoder training)
    + computed by xxx.m and put in score dir

## Running 

1. train and test neural encoders with bert representation in 5-fold cross validation,using 10 threads parallel.

> python encoder_roi.py --subject M01 --pooling bert --roi 0 --atlas 2 --jobs 10
   + the above arguments mean:
      + --subject subject id 
      + --pooling word or sentence embedding name
      + --roi roi id 
      + --atlas atlas id
      + --jobs number of parallel jobs