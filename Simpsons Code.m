% Initial Setup
categories = {'abraham_grampa_simpson','homer_simpson'};                                               % Name of all the categories you want to run
rootFolder = fullfile('C:\Temp\the-simpsons-characters-dataset\simpsons_dataset', 'simpsons_dataset'); % Set Root Folder
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');                 % Set image data store as only the predefined categories
tbl = countEachLabel(imds)                                                                             % Double check count of images in each foldernames
minSetCount = min(tbl{:,2});                                                                           % Determine the smallest amount of images in a category
imds = splitEachLabel(imds, minSetCount, 'randomize');                                                 % Set image folders to be the same size
countEachLabel(imds)                                                                                   % Triple check count of images in each foldernames

% Extract the first image from each folder and plot them to check
grampa = find(imds.Labels == 'abraham_grampa_simpson', 1);
homer = find(imds.Labels == 'homer_simpson', 1);
figure
subplot(1,3,1);
imshow(readimage(imds,homer))
subplot(1,3,2);
imshow(readimage(imds,grampa))

% Setup the transfer learning network
net = resnet50();                                                                                       

% Couple plots and define the NN and parameters
figure
plot(net)
title('First section of ResNet-50')
set(gca,'YLim',[150 170]);
net.Layers(1)
net.Layers(end)
numel(net.Layers(end).ClassNames)  % Number of class names for ImageNet classification task                                                

% Split the training test sets
[trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomize'); 

%Adjust the image size so that its readable by the classifier
imageSize = net.Layers(1).InputSize;
augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, 'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(imageSize, testSet, 'ColorPreprocessing', 'gray2rgb');

% Set NN weights
w1 = net.Layers(2).Weights; 
w1 = mat2gray(w1);
w1 = imresize(w1,5); 
figure
montage(w1)
title('First convolutional layer weights')
featureLayer = 'fc1000';

% Setup NN Parameters
trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns'); 

% Get training labels from the trainingSet
trainingLabels = trainingSet.Labels;

% Train multiclass SVM classifier using a fast linear solver, and set
% 'ObservationsIn' to 'columns' to match the arrangement used for training
% features.
classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

% Running the model on the test set and evaluating	
% Extract test features using the CNN
testFeatures = activations(net, augmentedTestSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

% Pass CNN image features to trained classifier
predictedLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

% Get the known labels
testLabels = testSet.Labels;

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2))

% Display the mean accuracy
mean(diag(confMat))

% Run the new model on a new image
newImage = imread(fullfile(rootFolder, 'airplanes', 'image_0690.jpg'));

% Create augmentedImageDatastore to automatically resize the image when
% image features are extracted using activations.
ds = augmentedImageDatastore(imageSize, newImage, 'ColorPreprocessing', 'gray2rgb');

% Extract image features using the CNN
imageFeatures = activations(net, ds, featureLayer, 'OutputAs', 'columns');

% Make a prediction using the classifier
label = predict(classifier, imageFeatures, 'ObservationsIn', 'columns')