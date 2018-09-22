% Automatically determine filepath using this script's location
S = dbstack();
filepath = erase(mfilename('fullpath'), S(1).name);

% Initial Setup
characters = {                                                                         % Uncomment all characters you want to run
    'homer_simpson'
    'ned_flanders'                                                                     % Top 10 characters (samples >1000)
    %'moe_szyslak'
    %'lisa_simpson'
    %'bart_simpson'
    %'marge_simpson'
    %'krusty_the_clown'
    %'principal_skinner'
    %'charles_montgomery_burns'
    %'milhouse_van_houten'
    }; 

trainingFolder = fullfile(filepath, 'simpsons_train_top10');                            % Set Training Folder
train_imds = imageDatastore(fullfile(trainingFolder, characters), 'LabelSource', 'foldernames');  % Set image data store as only the predefined characters
tbl = countEachLabel(train_imds)                                                        % Double check count of images in each foldernames
minSetCount = min(tbl{:,2});                                                            % Determine the smallest amount of images in a character
train_imds = splitEachLabel(train_imds, minSetCount, 'randomize');                      % Set image folders to be the same size
countEachLabel(train_imds)                                                              % Triple check count of images in each foldernames

% Extract the first image from each folder and display them to check
for n = 1:length(characters)
    character = find(train_imds.Labels == characters(n), 1);
    disp(characters(n))
    figure
    %subplot(1,3,n);    %PB note -> number of characters now exceeds number of subplots allowed
    imshow(readimage(train_imds,character))
end
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
[trainingSet, testSet] = splitEachLabel(train_imds, 0.3, 'randomize'); 

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

% Display the mean accuracy on test partition
mean(diag(confMat))

% Run the new model on a new image
newImage = imread(fullfile(trainingFolder, 'airplanes', 'image_0690.jpg'));

% Create augmentedImageDatastore to automatically resize the image when
% image features are extracted using activations.
ds = augmentedImageDatastore(imageSize, newImage, 'ColorPreprocessing', 'gray2rgb');

% Extract image features using the CNN
imageFeatures = activations(net, ds, featureLayer, 'OutputAs', 'columns');

% Make a prediction using the classifier
label = predict(classifier, imageFeatures, 'ObservationsIn', 'columns')


