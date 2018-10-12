# The Simpsons Character Classification

Image Processing GIT for 31256

This project demonstrates the use of a Support Vector Machine (SVM) trained with features via a Convolutional Neural Network (CNN) to classify Simpsons Characters. The Model was trained on 1400+ JPEG files of characters and generates an overall accuracy score for the overall number of classes trained. This project was developed in MATLAB using the ResNet-50 Deep Learning package, which had its final convolution and classification layers replaced to be specific to the Simpson's dataset. 
To date we have achieved an overall accuracy score of 95.3% accuracy for 10 characters.  
The user can elect to re-run this result without training or choose the number of characters (classes) to train.
 

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

#### To replicate this these results, run the algorithm 'as-is' without training. One can also refer to the 'saved_results' folder for output images.

## Authors

### Note
This was a team project with significant crossover in areas of work and equal contributions from all team members.  The primary roles, tasks and focus areas for each are detailed below:

* **Michael Aquilina 12004521** - *Documentation and Testing*     
	During the research proposal, I focused on writing about the Simpsons dataset and what input was going to be used in the model. 	Throughout the actual implementation of the project, I was initially tasked with testing the baseline model that was produced. 		The team planned to use GPU processing, however at the particular time of testing, this feature was not complete. I spent my 		time executing various different test cases from 2-10 classes using the following specifications:  
  	
	* NVidia 1080 GTX
	* 32Gb ram
	* Ryzen 1700 @ 3.00Ghz  
	  
	Using the baseline model, this achieved an average time of ~22 minutes. As well as testing the model, I prepared the readme file which contains the following sections:  the model prerequisites. Installing and supplying the Resnet package as an executable and Running the model.

* **Peter Bower 98136916** - *Development and Tuning*   
	During the project I have focused on building on the initial ResNet-50 transfer learning model. This included helping take it from 71% accuracy to 95% accuracy via replacing ResNet-50's final convolutional and classification layers with custom ones, and extracting CNN features from the Simpson's dataset. I also added a live Training Progress chart, montage of incorrectly classified images and confusion matrix to visualise accuracy, and setup auto-hyperparameter optimisation for the SVM. Finally, when our accuracy plateued I reviewed the incorrectly classified characters and added+labelled 300-400 more training images for the 4 characters with the smallest sample sizes, and retrained, bringing the final accuracy result to 95.3%.

* **Scott Casey 12032330** - *Documentation, Scalability and Testing*   
	As part of the group project I was involved primarily with coding scalability and prevention of errors when the model is being trained or run. The Scalability coding included things such as; prompting the user how many classes (characters) to train the model on which has a serious impact on performance and run time. Enabling the ability to adjust the train test split of the data. To prevent input errors which could crash the program I added in safe guards such as loops that required only valid inputs otherwise it would ask again. An example of this is if you tried to train 0 classes the code would have crashed, but now it instead asks the user for another input. In addition to this I also spent significant time testing the model and running different iterations and solving random errors we encounted. Finally, I contributed to the documentation of the group work.
	

* **Eric Chan** - *Documentation and Testing*  
	In the project requirements and specifications, I researched various methods that would also be viable in solving our classification problem, these include: Decision trees/ random forest, K-NN and adaptive boosting and reasons why they may or may not be a more viable option. I also assisted in editing the final document for submission. Using this research, I further tested these some of these methods on our dataset and compared these results with our ResNet-50 learning model. Furthermore, I assisted in testing the model and ensure it ran smoothly on my machine.
	

* **Elektra Neocleous 98098445** - *Documentation and Testing*  
	In this project I contributed to the documentation within the proposal and presentation and testing of the code. Within the proposal I researched into the image classification problem, the dataset sourced and algorithms we could apply to solve our problem. I discussed these points along with methodology of the machine learning process. I assisted in the editing and formatting of the document with the team. Additionally, I set up the presentation and split up the key points of discussions amongst members. In terms of code, I overlooked the preparation and creation process, assisting the team with idea generation in what we wanted executed. I assisted in testing the code and critiquing syntax. 
  
  	
* **Josh Overett 11719097** - *Development and Testing*   
	As part of the group project I found and documented 8 potential image data sets and explored viable techniques and packages for training them. After the group chose the Simpsons dataset I found a MATLAB guide from which I was able to build the first version of the training and testing code in MATLAB. Once the ResNet-50 package had been chosen and installed, I adjusted some parameters and ensured that the code could run bug free and had test stages throughout to display images and confirm the success of each of the previous steps. I also contributed to project documentation at various stages and did final testing and work on the evaluation.
	
 

## Source References

* Attia, Alex - Creator of the Simpsons Dataset - [Kaggle URL](https://www.kaggle.com/alexattia/the-simpsons-characters-dataset)
Mathworks - ResNet50 - Software Download - [ResNet50 Download](https://au.mathworks.com/matlabcentral/fileexchange/64626-deep-learning-toolbox-model-for-resnet-50-network)
* Kaggle - ResNet50 - [Overview of ResNet50](https://www.kaggle.com/keras/resnet50)


## Development References
[Train Deep Learning Network to Classify New Images- MATLAB & Simulink- MathWorks Australia](https://au.mathworks.com/help/deeplearning/examples/train-deep-learning-network-to-classify-new-images.html)

[Deep Learning in 10 lines of Matlab Code](https://blogs.mathworks.com/pick/2017/02/24/deep-learning-transfer-learning-in-10
