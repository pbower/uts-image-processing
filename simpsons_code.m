% Automatically determine filepath using this script's location
S = dbstack();
filepath = erase(mfilename('fullpath'), S(1).name);

%Prompt the user for imput of number of characters to train the model on
%This is for scalability on the users processing power and time avaliable
prompt = 'How many Characters(Classes) would you like to use in the creation of the model? Please enter a number between 2 and 10. This will impact time needed to train the model. ';
CharNo = input(prompt);
CharRev = 10 - CharNo;

% Initial Setup
characters = {                                                                         % Uncomment all characters you want to run
    'homer_simpson'
    'ned_flanders'                                                                     % Top 10 characters (samples >1000)
    'moe_szyslak'
    'lisa_simpson'
    'bart_simpson'
    'marge_simpson'
    'krusty_the_clown'
    'principal_skinner'
    'charles_montgomery_burns'
    'milhouse_van_houten'
    };
if CharNo <= 9
    for n = CharNo+1:10
        characters{n,1} = [];
        characters{n,:} = [];
    end
else
    
end
%characters = characters(end-8: 1,:);
 characters(cellfun('isempty',characters)) = [];

trainingFolder = fullfile(filepath, 'simpsons_train_top10');                            % Set Training Folder
train_imds = imageDatastore(fullfile(trainingFolder, characters), 'LabelSource', 'foldernames');  % Set image data store as only the predefined characters
tbl = countEachLabel(train_imds);                                                        % Double check count of images in each foldernames
minSetCount = min(tbl{:,2});                                                            % Determine the smallest amount of images in a character
train_imds = splitEachLabel(train_imds, minSetCount, 'randomize');                      % Set image folders to be the same size
countEachLabel(train_imds)                                                              % Triple check count of images in each foldernames




% Extract the first image from each folder and create a 2x5 box to display them
figure
hold on
for n = 1:length(characters)
    character = find(train_imds.Labels == characters(n), 1);
    disp(characters(n))
    subplot(2,5,n); 
    imshow(readimage(train_imds,character))
end
hold off
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


%{
PB COMMENT 22/9: The below section reuses similar code to the top part 
so we could look at splitting the file into classes and functions once we have everything working 
%}

% Evaluate model on 'Real' data
evalFolder = fullfile(filepath, 'simpsons_eval_top10');              % Set 'real' data Folder
evalSet = imageDatastore(fullfile(evalFolder, characters), 'LabelSource', 'foldernames');  % Set image data store as only the predefined characters
tbl = countEachLabel(evalSet)                                        % Double check count of images in each foldernames
minSetCount = min(tbl{:,2});                                         % Determine the smallest amount of images in a character
evalSet = splitEachLabel(evalSet, minSetCount, 'randomize');         % Set image folders to be the same size
countEachLabel(evalSet)  

% Check Evaluation Images Exist
for n = 1:length(characters)
    character = find(evalSet.Labels == characters(n), 1);
    disp(characters(n))
    figure
    imshow(readimage(evalSet,character))
end

%Adjust the evaluation image size so that its readable by the classifier
augmentedEvalSet = augmentedImageDatastore(imageSize, evalSet, 'ColorPreprocessing', 'gray2rgb');

% Running the model on the test set and evaluating	
% Extract test features using the CNN
evalFeatures = activations(net, augmentedEvalSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

% Pass CNN image features to trained classifier
predictedLabels = predict(classifier, evalFeatures, 'ObservationsIn', 'columns');

% Get the known labels
evalLabels = evalSet.Labels;

% Tabulate the results using a confusion matrix.
confMat = confusionmat(evalLabels, predictedLabels);

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2))

% Display the mean accuracy on evaluation partition
mean(diag(confMat))

% Display Heatmap of Results
figure
[cmat, classNames] = confusionmat(evalLabels, predictedLabels);
h = heatmap(classNames,classNames,cmat);
xlabel('Predicted Class');
ylabel('True Class');
title('Confusion Matrix');

% Display Confusion Matrix
figure
title('Confusion Matrix');
plotconfusion(evalLabels,predictedLabels);

% Display some test images with predicted classes and probabilities
figure
sgtitle('Incorrectly Labelled Images')
hold on
count = 0
for n = 1:length(evalLabels)
    if evalSet.Labels(n) ~= predictedLabels(n)
        title(n)
        count = count + 1;
        subplot(15,15,count); 
        imshow(readimage(evalSet,n))
    end
end
hold off


