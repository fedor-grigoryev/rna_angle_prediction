# RNA bioinformatics - M2-GENIOMHE project 

This code is a skeleton for your project. 

Please start from this your project. 

## Repository

The folder is composed of: 
- `data`: a folder with the training (`data/TrainingSet`), testing (`data/TestSet`), sample (`data/sample`) folders and `SPOT-RNA-1D` folder.
        The `sample` folder contains a `.fasta` file that should be used to do inference for the delivery. 
        The `SPOT-RNA-1D` folder contains the predictions from `SPOT-RNA-1D` for the Training set (`data/SPOT-RNA-1D/training.json`) and Test set (`data/SPOT-RNA-1D/test.json`).
- `lib`: a folder where should be cloned a code that computes the dihedral angles. Please fork this [repository](https://github.com/EvryRNA/rna_angles_prediction_dssr/tree/main)
- `src`: a folder where you should put all your implementations. 

You should at the end also include python requirements to run your code with good library versions. 
You should also delete the content of the README by documentation of your implementation and code details. 
