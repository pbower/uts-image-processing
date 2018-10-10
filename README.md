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
Dataset - The Simpsons dataset must be present in the root of the codebase folder directory. 
This will download and work automatically provided the project is cloned or downloaded via the GIT .zip file.
```

### Installing

The first step in order to execute the classification model is installing the required MATLAB package.

This package will be provided as an exe in the root of the project folder.

Installing Resnet

```
From MATLAB

Home > Project folder > resnet50.mlpkginstall

MATLAB will open an install prompt for "Neural Network Toolbox Model for the latest ResNet-50 Network Version applicable to your Matlab version."

Tick the checkbock, click next and follow the prompts to complete the installation.
```

## Running the Model

This section will explain how to run the classification model.

### Execute the classification model

From Matlab Home:

1. Open the simpsons_code.m file
2. From Editor, Click Run
3. You will prompted as to whether you would like to re-train the model. If you do so, you will then be able to choose between 2-10 characters which will impact training time. 
<em>Note: Training time for 10 characters was approximately 2 hours for 5 epochs on a 2.9 GHz Intel Core i7 2017 MacBook Pro with 16GB of RAM and no CUDA GPU. </em>.  
4. Training includes replacing the final ResNet-50 layers with convolutional and classification feature layers trained from the Simpsons dataset.
5. These features are then used to train an SVM Classifier which does the actual classification in the model.

The model will now execute and begin training/scoring. Without training, the model takes approximately 2 minutes to run and produce a score. If training, our team recommends running it on '2' characters for the first run as each system time varies depending on CPU power.

## Built With

*  MATLAB - MATLAB (matrix laboratory) is a multi-paradigm numerical computing environment and proprietary programming language developed by MathWorks.
*  Resnet50 -  Matlab's Deep Learning Toolbox Model for ResNet-50

## Major Optimisation Steps Taken:	<em>Note: Accuracy %'s are with respect to 10 characters</em>
1. Initial Algorithm Development (Res-Net 50 > SVM Classifier Model)  -> ~71% Acc.
2. Replace final Convolutional & Classification layers with custom ones to train on Simpson's features - > ~91% Acc.
3. Adjust training maxEpochs to avoid overfitting -> ~93% Acc.
4. Add self-learning hyperparameters for SVM -> minor improvement < ~0.5%
5. Try training with different learning rates and parameters -> no improvement and in some instances training failed to converge; optimal learning rate 0.001

<em> At this stage we further reviewed misclassified Training examples and realised they were actually characters who had more training data (e.g. Homer, Bart), so we needed to 'top up' our training volumes for characters who had the least but the model had no issues with (e.g. Milhouse).</em>

5. Create 300-400 more training examples via:
	a) Running a python script to automatically download 100 images of the characters with the lowest training sample sizes from Google Images (included in Dev Tools folder)
	b) For images that needed more training data than Google had available (e.g. Milhouse), resample them by flipping them on the horizontal axis using Bash
	c) Organise them into the image naming conventions using Bash
6. Re-Ran training on several parameters. Original ones produced the best results -> 95.3% Acc.

## Authors

* **Elektra Neocleous** - *Documentation and Testing*  
	Insert contribution here.
  
* **Eric Chan** - *Documentation and Testing*  
	Insert contribution here. 
  	
* **Josh Overett** - *Documentation and Testing*   
	Insert contribution here.
  	
* **Michael Aquilina** - *Documentation and Testing*   
	Insert contribution here.
  	
* **Peter Bower** - *Development and Tuning*   
	Insert contribution here.
  
* **Scott Casey** - *Documentation and Scalability*   
	Insert contribution here.
	

## Source References

* Attia, Alex - Creator of the Simpsons Dataset - [Kaggle URL](https://www.kaggle.com/alexattia/the-simpsons-characters-dataset)
Mathworks - RestNet 50 - Software Download - [ResNet50 Download](https://au.mathworks.com/matlabcentral/fileexchange/64626-deep-learning-toolbox-model-for-resnet-50-network)
* Kaggle - Resnet50 - [Overview of ResNet50](https://www.kaggle.com/keras/resnet50)


## Development References
[Train Deep Learning Network to Classify New Images- MATLAB & Simulink- MathWorks Australia](https://au.mathworks.com/help/deeplearning/examples/train-deep-learning-network-to-classify-new-images.html)

[Deep Learning in 10 lines of Matlab Code](https://blogs.mathworks.com/pick/2017/02/24/deep-learning-transfer-learning-in-10
