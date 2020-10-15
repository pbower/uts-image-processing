# The Simpsons Character Classification

Image Processing GIT for 31256

This project demonstrates the use of a Support Vector Machine (SVM) trained with features via a Convolutional Neural Network (CNN) to classify Simpsons Characters. The Model was trained on 1400+ JPEG files of characters and generates an overall accuracy score for the overall number of classes trained. This project was developed in MATLAB using the ResNet-50 Deep Learning package, which had its final convolution and classification layers replaced to be specific to the Simpson's dataset. 
To date we have achieved an overall accuracy score of 95.3% accuracy for 10 characters.  
The user can elect to re-run this result without training or choose the number of characters (classes) to train.
 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for testing purposes.

### Prerequisites

There are four prerequisites in order to run this model.

```
MATLAB - Downloaded and Installed
Deep Learning Toolbox - Downloaded and Installed
Res-Net50 Matlab CNN - Downloaded and Installed (included with instructions below)
Dataset - The Simpsons dataset must be present in the root of the codebase folder directory. 
This will download and work automatically provided the project is cloned or downloaded via the GIT .zip file.
```

### Installing Res-Net 50

The first step in order to execute the classification model is installing the Res-Net 50 MATLAB package.

Installing Resnet:

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
*  ResNet50 -  Matlab's Deep Learning Toolbox Model for ResNet-50

## Major Optimisation Steps Taken:	
#### Note: Accuracy %'s are with respect to 10 characters
1. Initial Algorithm Development (ResNet-50 > SVM Classifier Model)  <em>~71% Acc.</em>
2. Replace final Convolutional & Classification layers with custom ones to train on Simpson's features <em>~91% Acc.</em>
3. Adjust training maxEpochs to avoid overfitting <em>~93% Acc.</em>
4. Add self-learning hyperparameters for SVM -> minor improvement <em>+ ~0.5% Acc. </em>
5. Try training with different learning rates and parameters -> no improvement and in some instances training failed to converge; optimal learning rate 0.001

<em> At this stage we further reviewed misclassified Training examples and realised they were actually characters who had more training data (e.g. Homer, Bart), so we needed to 'top up' our training volumes for characters who had the least but the model had no issues with (e.g. Milhouse).</em>

5. Create 300-400 more training examples via:  
	* Running a python script to automatically download 100 images of the characters with the lowest training sample sizes from Google Images <em>(included in Dev Tools folder)</em>
	* For images that needed more training data than Google had available (e.g. Milhouse), resample them by flipping them on the horizontal axis using Bash
	* Organise them into the image naming conventions using Bash
6. Re-ran training on several parameters. Original ones produced the best results -> <em>95.3% Acc.</em>

Note: After this we attempted a custom 'Weighted Classification Layer' model to fully utilise the number of training samples for each character. This produced no improvement (95.1% Acc.).

#### To replicate this these results, run the algorithm 'as-is' without training. One can also refer to the 'saved_results' folder for output images.

## Source References

* Attia, Alex - Creator of the Simpsons Dataset - [Kaggle URL](https://www.kaggle.com/alexattia/the-simpsons-characters-dataset)
Mathworks - ResNet50 - Software Download - [ResNet50 Download](https://au.mathworks.com/matlabcentral/fileexchange/64626-deep-learning-toolbox-model-for-resnet-50-network)
* Kaggle - ResNet50 - [Overview of ResNet50](https://www.kaggle.com/keras/resnet50)


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
