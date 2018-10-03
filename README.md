# The Simpsons Character Classification

Image Processing GIT for 31256

This project demonstrates the use of a Support Vector Machine (SVM) training with features from a Convoluted Neural Network (CNN)
to classify the top 10 Simpsons Characters. The Model will loop through various JPEGS of characters and generate a score. It will
return a overall accuracy for the number of classes trained. This project was developed in MATLAB using the Resnet package.
 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for testing purposes.

### Prerequisites

There are two prerequisites in order to run this model.

```
MATLAB - Downloaded and Installed
Dataset - The Simpsons dataset downloaded and present in the root of the codebase folder directory.
```

### Installing

The first step in order to execute the classification model is installing the required MATLAB package.

This package will be provided as an exe in the root of the project folder.

Installing Resnet

```
From MATLAB

Home > Project folder > install_supportsoftware.exe

MATLAB will open an install prompt for "Neural Network Toolbox Model for ResNet-50 Network Version 18.1.0"

Tick the checkbock, click next and follow the prompts to complete the installation.
```

## Running the Model

This section will explain how to run the classificaiton model.

### Execute the classification model

From Matlab Home:

1. Open the simpsons_code.m file
2. From Editor, Click Run
3. You will prompted by the following message in MATLAB: <em>How many Characters(Classes) would you like to use in the creation of the model? Please enter a number between 2 and 10. This will impact time needed to train the model.</em>.  
3. Enter a number from 2-10 in the MATLAB commandline.

The model will not execute and begin training/scoring. This time varies from ~3 to 15 minutes depending on the number
of classes given. Our team recommends giving '2' for the first run to determine how long it will take your system to run
the model as it varies for different systems.


## Built With

*  MATLAB - MATLAB (matrix laboratory) is a multi-paradigm numerical computing environment and proprietary programming language developed by MathWorks.
*  Resnet50 -  ResNet-50 Pre-trained Model for Keras


## Authors

* **Elektra Neocleous** - *Documentation and Testing* 
* **Eric Chan** - *Documentation and Testing* 
* **Michael Aquilina** - *Documentation and Testing* 
* **Scott Casey** - *Documentation and Scalability* 
* **Peter Bower** - *Development and Tuning* 
* **Josh Overett** - *Documentation and Testing* 


## References

* Reference the github that prepared the data.
* Reference Resnet50
* etc

