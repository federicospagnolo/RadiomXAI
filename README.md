<h1 align="center">RadiomXAI </h1>

> Scripts to improve classification results, exploiting radiomic features of XAI maps (e.g., saliency).<br /> `RadiomXAI` can significantly boost performances of machine learning networks, which operate on medical imaging (tested on MRI), by increasing the F1 score.

## 🚀 Usage

First, make sure you have python >=3.9 installed.

To build the environment, an installation of conda or miniconda is needed. Once you have it, please use
```sh
conda env create -f environment.yml
```
to build the tested environment using the provided `environment.yml` file. 

The `generate_csv.py` uses XAI maps to create a `.csv` table, which contains the class, filename, and radiomic features of each XAI example.
The variable `path` needs to be modified according to the path containing XAI data. Data should be organized so that XAI maps on true positive examples are inside a folder "TP", and false positive examples inside "FP". 

The script `bootstrap_radiomics_classifier.py` exploits the extracted radiomic features to feed a linear classifier (tested on logistic regression) to refine the classification of true positive and false positive examples.

## Code Contributors

This work is part of the project MSxplain, and has been published in MICCAI2024 proceedings (iMIMIC workshop).

## Author

👤 **Federico Spagnolo**

- Github: [@federicospagnolo](https://github.com/federicospagnolo)
