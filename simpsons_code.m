% Automatically determine filepath using this script's location
S = dbstack();
filepath = erase(mfilename('fullpath'), S(1).name);
charNo = textread('lastTrainedCharNo.txt');

% Prompt user for CNN feature and SVM classifier training
if exist('teamClassifier.mat', 'file') == 2 && exist('teamNet.mat', 'file') == 2
    disp(strcat(['Existing model with ', num2str(charNo), ' trained characters found...']))
    prompt = 'Would you like to re-train?';
    train = input(prompt, 's');
    if train == 'n'
        disp('Running classifier without re-training.')
    end 
else 
    disp('Unable to teamClassifier.mat and/or teamNet.mat.')
    disp('These files need to be added to the root folder or else complete the training process which will automatically output these files to the directory.')
    train = 'y';
end
 if train == 'y'
    % Prompt the user for imput of number of characters to train the model on
    % This is for scalability on the users processing power and time avaliable
    prompt = 'How many Characters(Classes) would you like to use in the creation of the model? Please enter a number between 2 and 10. This will impact time needed to train.';
    charNo = input(prompt);
    train = 'y'; % Automatically re-retrain the SVM if features are re-trained
    disp('Training process will replace ResNet-50 fc_1000 and classification 2layers and SVM Classifier will then re-train')
 end
 
% Initial Setup
characters = {                                                                         
    'homer_simpson'
    'ned_flanders'                                                                     
    'moe_szyslak'
    'lisa_simpson'
    'bart_simpson'
    'marge_simpson'
    'krusty_the_clown'
    'principal_skinner'
    'charles_montgomery_burns'
    'milhouse_van_houten'
    };

characters = characters(1:charNo);

% Train Neural Network Features
if train == 'y' 
    
    trainingFolder = fullfile(filepath, 'simpsons_train_top10');                            % Set Training Folder
    train_imds = imageDatastore(fullfile(trainingFolder, characters), 'LabelSource', 'foldernames');  % Set image data store as only the predefined characters
    tbl = countEachLabel(train_imds);                                                        % Double check count of images in each foldernames
    minSetCount = min(tbl{:,2});                                                            % Determine the smallest amount of images in a character
    train_imds = splitEachLabel(train_imds, minSetCount, 'randomize');                      % Set image folders to be the same size
    countEachLabel(train_imds)                                                              % Triple check count of images in each foldernames

    % Extract the first image from each folder and create a 2x5 box to display them
    figure
    if version >= 9.5
        sgtitle('First image from each character folder')
    end
    hold on
    for n = 1:length(characters)
        character = find(train_imds.Labels == characters(n), 1);
        disp(characters(n))
        subplot(2,5,n); 
        imshow(readimage(train_imds,character))
    end
    hold off

    % Split the training and test sets
    [trainingSet, testSet] = splitEachLabel(train_imds, 0.3, 'randomize'); 

    % Adjust the image size so that its readable by the classifier
    net = resnet50();
    imageSize = net.Layers(1).InputSize;

    augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, 'ColorPreprocessing', 'gray2rgb');
    augmentedTestSet = augmentedImageDatastore(imageSize, testSet, 'ColorPreprocessing', 'gray2rgb');

    % Setup the transfer learning network    
    analyzeNetwork(net)
    lgraph = layerGraph(net);

    % Set Training Options
    options = trainingOptions('sgdm', 'InitialLearnRate', 0.001, 'MaxEpochs', 3, 'MiniBatchSize', 64, 'Shuffle','every-epoch','Plots','training-progress');

    % Check that the last 2 layer replacements are connected correctly
    figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    title('Check Last 2 Replacement Layers')
    plot(lgraph)
    ylim([0,10])

    % Couple plots and define the NN and parameters
    figure
    plot(net)
    title('First section of ResNet-50')
    set(gca,'YLim',[150 170]);
    net.Layers(1)
    net.Layers(end)

    numel(net.Layers(end).ClassNames)  % Number of class names for ImageNet classification task    

    layers = net.Layers

    % Freeze all the old ResNet weights to avoid retraining the entire model

    for i = 1:numel(layers)-3
        if isprop(layers(i),'WeightLearnRateFactor')
            layers(i).WeightLearnRateFactor = 0;
        end
        if isprop(layers(i),'WeightL2Factor')
            layers(i).WeightL2Factor = 0;
        end
        if isprop(layers(i),'BiasLearnRateFactor')
            layers(i).BiasLearnRateFactor = 0;
        end
        if isprop(layers(i),'BiasL2Factor')
            layers(i).BiasL2Factor = 0;
        end
    end

    % Replace the last training layer with our own
    numClasses = numel(categories(train_imds.Labels));
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
            'Name','fc1000', ...
            'WeightLearnRateFactor',10, ...
            'BiasLearnRateFactor',10);
    lgraph = replaceLayer(lgraph,'fc1000',newLearnableLayer);

    % Replace the Classification Layer with a new one
    newClassLayer = classificationLayer('Name','ClassificationLayer_fc1000');
    lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000',newClassLayer);
    connections = lgraph.Connections;

    % Our Custom Neural Net with the last 2 layers
    net = trainNetwork(augmentedTrainingSet, lgraph, options);
    %Save Neural Network Weights
    teamNet = net;
    save ('teamNet.mat', 'teamNet')
end

% Can be commented out to run ResNet-50 without the last layers replaced
load teamNet
net = teamNet();
imageSize = net.Layers(1).InputSize;

% Set NN weights
w1 = net.Layers(2).Weights; 
w1 = mat2gray(w1);
w1 = imresize(w1,5); 
figure
montage(w1)
featureLayer = 'fc1000';
title('First convolutional layer weights');

if train == 'y'
    % Setup NN Feature Parameters
    trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, ...
        'MiniBatchSize', 64, 'OutputAs', 'columns'); 

    % Get training labels from the trainingSet
        trainingLabels = trainingSet.Labels;

    % Train multiclass SVM classifier using a fast linear solver, and set
    % 'ObservationsIn' to 'columns' to match the arrangement used for training
    % features.
    classifier = fitcecoc(trainingFeatures, trainingLabels, ...
        'OptimizeHyperparameters','auto', ...
        'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
        'expected-improvement-plus'),...
        'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');
    teamClassifier = classifier;
    save('teamClassifier.mat', 'teamClassifier')
    
    % Running the model on the test set and evaluating	
    % Extract test features using the CNN
    testFeatures = activations(net, augmentedTestSet, featureLayer, ...
        'MiniBatchSize', 64, 'OutputAs', 'columns');

    % Pass CNN image features to trained classifier
    predictedLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

    % Get the known labels
    testLabels = testSet.Labels;

    % Tabulate the results using a confusion matrix.
    confMat = confusionmat(testLabels, predictedLabels);

    % Convert confusion matrix into percentage form
    confMat = bsxfun(@rdivide,confMat,sum(confMat,2))

    % Display the mean accuracy on test partition
    disp('Printing Test Accuracy:')
    mean(diag(confMat))
    
    % Write the number of trained characters to a text file
    fileID = fopen('lastTrainedCharNo.txt','w');
    fprintf(fileID,'%g\n', charNo);
    fclose(fileID); 
else
    load ('teamClassifier.mat', 'teamClassifier')
    classifier = teamClassifier;
end

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
    'MiniBatchSize', 64, 'OutputAs', 'columns');

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
title('Heatmap');

% Display Confusion Matrix
figure
title('Confusion Matrix');
plotconfusion(evalLabels,predictedLabels);

 % Display Montage of Incorrectly Labelled Images
 totalIncorrect = sum(evalSet.Labels ~= predictedLabels);
 incCount = 0;
 incorrectImageList = cell(1, totalIncorrect);
 for n = 1:length(evalLabels)
     if evalSet.Labels(n) ~= predictedLabels(n)
         incorrectImageList{incCount+1} = readimage(evalSet, n);
         incCount = incCount + 1;
     end
 end
 figure
 montage(incorrectImageList)
 title('Incorrectly labelled images')