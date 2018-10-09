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

[Deep Learning in 10 lines of Matlab Code](https://blogs.mathworks.com/pick/2017/02/24/deep-learning-transfer-learning-in-10-lines-of-matlab-code/)

[Image Category Classification Using Deep Learning](https://www.mathworks.com/examples/matlab-computer-vision/mw/vision-ex77068225-image-category-classification-using-deep-learning#17)

[Train neural network for deep learning - MATLAB trainNetwork- MathWorks Australia](https://au.mathworks.com/help/deeplearning/ref/trainnetwork.html#bu6sn60-2)

[Parameter Optimisation References](https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10)

[Compute convolutional neural network layer activations - MATLAB activations- MathWorks Australia](https://au.mathworks.com/help/deeplearning/ref/activations.html)

[Options for training deep learning neural network - MATLAB trainingOptions- MathWorks Australia](https://au.mathworks.com/help/deeplearning/ref/trainingoptions.html)

[2-D convolutional layer - MATLAB- MathWorks Australia](https://au.mathworks.com/help/deeplearning/ref/nnet.cnn.layer.convolution2dlayer.html#mw_308497ca-95e8-402a-9c40-18157b47a74a)

[Fully connected layer - MATLAB- MathWorks Australia](https://au.mathworks.com/help/deeplearning/ref/nnet.cnn.layer.fullyconnectedlayer.html#mw_bac20c29-f95b-423a-816a-428fc8e0463e)

[Create Simple Deep Learning Network for Classification- MATLAB & Simulink Example- MathWorks Australia](https://au.mathworks.com/help/deeplearning/examples/create-simple-deep-learning-network-for-classification.html)

[Pretrained Convolutional Neural Networks- MATLAB & Simulink- MathWorks Australia](https://au.mathworks.com/help/deeplearning/ug/pretrained-convolutional-neural-networks.html)

[Transfer Learning Using AlexNet- MATLAB & Simulink- MathWorks Australia](https://au.mathworks.com/help/deeplearning/examples/transfer-learning-using-alexnet.html)

[Classify Image Using GoogLeNet- MATLAB & Simulink- MathWorks Australia](https://au.mathworks.com/help/deeplearning/examples/classify-image-using-googlenet.html)

[Transfer Learning with Deep Network Designer- MATLAB & Simulink- MathWorks Australia](https://au.mathworks.com/help/deeplearning/ug/transfer-learning-with-deep-network-designer.html)

 [Deep Learning with Images- MATLAB & Simulink- MathWorks Australia](https://au.mathworks.com/help/deeplearning/deep-learning-with-images.html)
 
[Train an image category classifier - MATLAB trainImageCategoryClassifier- MathWorks Australia](http://au.mathworks.com/help/vision/ref/trainimagecategoryclassifier.html)
