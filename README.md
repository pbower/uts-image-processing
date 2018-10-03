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

From Matlab Home, open the simpsons_code.m file and run the program.

You will be prompted to provide the number of classes that you wish to train/model. This is provided as varying
systems take differing times depending on the number of classes. We recommend to test your system using 2 classes 
as this generally takes ~3 - 5 minutes.


## Built With

*  MATLAB - The web framework used
*  Resnet50 -  Dependency Management


## Authors

* **Elektra Neocleous** - *Documentation and Testing* 
* **Eric Chan** - *Documentation and Testing* 
* **Michael Aquilina** - *Documentation and Testing* 
* **Scott Casey** - *Documentation and Scalability* 
* **Peter Bowler** - *Development and Tuning* 
* **Josh Overett** - *Documentation and Testing* 


## References

* Reference the github that prepared the data.
* Reference Resnet50
* etc

