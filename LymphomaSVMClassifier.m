%% Lymphoma SVM Multi-Class Classifier
%% Samuel Ndirangu 
% Github: <https://github.com/SamNdirangu/ https://github.com/SamNdirangu/>
% 
% LinkedIn: <https://www.linkedin.com/in/samndirangu https://www.linkedin.com/in/samndirangu>
% 
% Date: 29/04/2019
% 
% *Introduction*
% 
% This dataset features 148 samples each with 18 features. This dataset was 
% introduced to train models to detect 3 cases within a lymphogram. Samples are 
% classified into four classes; normal, malignant, metastases and fibrosis. 
% 
% *References and Dataset source * 
% 
% I. Kononenko and B. Cestnik (1988). UCI Machine Learning Repository [ <https://archive.ics.uci.edu/ml/datasets/Lymphography 
% https://archive.ics.uci.edu/ml/datasets/Lymphography> ]. Irvine, CA: University 
% of California, School of Information and Computer Science. 
%% Step 1: Dataset Preparation
% Here we prepare our data so that it can be fed to the svm accordingly.
% 
% The data is first loaded from a given xlsx file and the necessry collumns 
% extracted.
% 
% The data is then split into training and tests sets. One trains the svm 
% the other is used to evaluate its accuracy perfomance.
%% File Upload
%%
%The prefix ckd refers to Chronic Kidney Disease
%Stores our file name and path
[file,path] = uigetfile({'*.xlsx';'*.csv'},'Select Data File')
selectedfile = fullfile(path,file);
opts = detectImportOptions(selectedfile); %Auto Load options for it
temp_data = readtable(selectedfile,opts); %Read the xlsx file and store it.
data = table2cell(temp_data); %Convert our table to a cell array

%Get our feature names
data_feature_names = cellstr(strrep(temp_data.Properties.VariableNames,'_',' '));

sample_no = size(data,1) %Get the number of all diagnosis
%Create our X and Y for SVM Training
X_temp = cell2mat(data(:,2:19));
%Our Y or dependent varible is in collumn 1
Y_temp = cell2mat(data(:,1));

rand_num = randperm(148); % Permutate numbers radomly, This works as shuffling our rows

X = X_temp(rand_num(1:end),:);
Y = Y_temp(rand_num(1:end),:);

%Get the number of features of X
noFeatures = size(X,2);
%% Step 2: Preparing validation set out of trainin set (K-fold CV)
% To reduce any underfitting or overfitting that may occur during testing the 
% data is cross validated using K-Fold 5 times. The folds are stratified ensuring 
% a uniform distribution of each class with in each fold further lowering any 
% data bias that would have been present.
%%
%Validate now the training set using Kfold 5 times
%A warning will however be thrown as one class is rare and cannot be present in all folds
CV = cvpartition(Y,'KFold',5,'Stratify',true);
%% Step 3: Feature Ranking
% This steps ranks our feature and creates a feature sets consisting  a given 
% number of all the features from 1 feature to all features of the dataset
% 
% We use sequential FS and loop it a number of times equal to the number 
% of features filling in a logical matrix of features.
%%
opts = statset('display','iter','UseParallel',true); %Sets the display option
rng(5); %This sets our random state.

%We create a variable to store our template SVM option that the ecoc function will use to train multi-class SVM models
svmTemplate = templateSVM('Standardize',false,'KernelFunction','gaussian');

fun = @(train_data, train_labels, test_data, test_labels)...
    sum(predict(fitcecoc(train_data, train_labels,'Learners',svmTemplate), test_data) ~= test_labels);

%Rank our features using Sequential FS forward selection
%Inside history we store the ranking of features
[fs, history] = sequentialfs(fun, X, Y, 'cv', CV, 'options', opts,'nfeatures',noFeatures);
%% Step 4: Kernel and Feature Selection
% This step now analyzes each kernel functions performance in regard to a given 
% feature set
%%
rng(3);
ave_Accuracy(noFeatures,6) = 0; %Initializes where we'll store our performance for each feature set and kernel
for count=1:noFeatures
    %MStore our best features
    ave_Accuracy(count,1) = count;
    
    %using a Linear kernel
    template = templateSVM(...
        'KernelFunction', 'linear', ...
        'KernelScale', 'auto');
    lym_Model = fitcecoc(X(:,history.In(count,:)),Y,'Learners', template,'Coding','onevsone','CVPartition',CV);
    % Compute validation accuracy
    ave_Accuracy(count,2) = (1 - kfoldLoss(lym_Model, 'LossFun', 'ClassifError'))*100;
    
    %using a RBF kernel
    template = templateSVM(...
        'KernelFunction', 'rbf', ...
        'KernelScale', 'auto');
    lym_Model = fitcecoc(X(:,history.In(count,:)),Y,'Learners', template,'Coding','onevsone','CVPartition',CV);
    % Compute validation accuracy
    ave_Accuracy(count,3) = (1 - kfoldLoss(lym_Model, 'LossFun', 'ClassifError'))*100;
    
    %using a Gaussian kernel
    template = templateSVM(...
        'KernelFunction', 'gaussian', ...
        'KernelScale', 'auto', ...
        'BoxConstraint', 1, ...
        'Standardize', true);
    lym_Model = fitcecoc(X(:,history.In(count,:)),Y,'Learners', template,'Coding','onevsone','CVPartition',CV);
    % Compute validation accuracy
    ave_Accuracy(count,4) = (1 - kfoldLoss(lym_Model, 'LossFun', 'ClassifError'))*100;
    
    %using a polynomial kernel
    template = templateSVM(...
        'KernelFunction', 'polynomial',...
        'KernelScale', 'auto', ...
        'BoxConstraint', 1, ...
        'Standardize', true);
    lym_Model = fitcecoc(X(:,history.In(count,:)),Y,'Learners', template,'Coding','onevsone','CVPartition',CV);
    % Compute validation accuracy
    ave_Accuracy(count,5) = (1 - kfoldLoss(lym_Model, 'LossFun', 'ClassifError'))*100;
    
    %using a polynomial kernel
    template = templateSVM(...
        'KernelFunction', 'polynomial', ...
        'PolynomialOrder',3, ...
        'KernelScale', 'auto', ...
        'BoxConstraint', 1, ...
        'Standardize', true);
    lym_Model = fitcecoc(X(:,history.In(count,:)),Y,'Learners', template,'Coding','onevsone','CVPartition',CV);
    % Compute validation accuracy
    ave_Accuracy(count,6) = (1 - kfoldLoss(lym_Model, 'LossFun', 'ClassifError'))*100;
end
%% Visualize Results
%%
figure
plot(ave_Accuracy(:,2:6))
title('Lymphography Model Perfomance against No. of Features')
xlabel('Number of Features')
ylabel('Model Perfomance')
legend('Linear','RBF','Gaussian','Quadratic','Cubic')
grid on;
%% Step 5: Model Selection: Best Hyperparameters
% We now need to search for the best hyperparameters for the highest accuracy 
% perfomance for the dataset.The following functions will be utilized to tune 
% the SVM parameters C and Kernel function variables We use fitcsvm and train 
% using the best features obtained from step 4. The kernel function is selected 
% from the best perfoming Kernel in the previous step
%%
%The best observed Kernel is Linear
template = templateSVM(...
    'KernelFunction', 'gaussian', ...
    'KernelScale', 'auto', ...
    'Standardize', true);
rng(3);
%For the bayesain optimizer we let it use the defaults but increase its Maximum Objective evaluations to 30
SVM_Model = fitcecoc(X(:,history.In(17,:)),Y,'Learners', template,...
    'OptimizeHyperparameters','auto','Coding','onevsone','HyperparameterOptimizationOptions',struct('UseParallel',true,...
    'ShowPlots',false,'MaxObjectiveEvaluations',30,'Repartition',true));

%% Step 6: Train a SVM and Evaluate Validation Loss : Kernel linear
% This step will now train the SVM using the best features selected in step 
% 3 and the optimal hyperparameters found in step 4.
%%
%Pass on our X and Y to the SVM classifier.
rng(3); %seeds our random generator allowing for reproducibility of results
%Box Costraint is our penalty factor C, the higher the stricter the hyperplane
constraint = 893.62;
kernelScale = SVM_Model.BinaryLearners{1}.KernelParameters.Scale;

%using a Linear kernel
template = templateSVM(...
    'BoxConstraint',constraint,...
    'KernelScale',kernelScale,...
    'KernelFunction','polynomial',...
    'Standardize', false);

%using a polynomial kernel
best_SVMModel = fitcecoc(X(:,history.In(17,:)),Y,'CVPartition',CV,'Coding','ternarycomplete');


%This averages our cross validated models loss. The lower the better the prediction
% Compute validation accuracy
SVM_ValidationAccuracy = (1 - kfoldLoss(best_SVMModel, 'LossFun', 'ClassifError'))*100
%This averages our cross validated models loss. The lower the better the prediction
bc_ModelLoss = kfoldLoss(best_SVMModel)
%% Step 7: Evaluate the model's perfomance using the test set.
%%
[Y_pred, validationScores] = kfoldPredict(best_SVMModel);
conMat=confusionmat(Y,Y_pred);
%Generate a Confusion Matrix
%The confusion matrix gives us a view of the rate of true positives to false negatives.
%It enables to have proper view of how effecive our model is in prediction of illness minimizing of having false negatives
conMatHeat = heatmap(conMat,'Title','Confusion Matrix Lmyphoma Disease','YLabel','Actual Diagnosis','XLabel','Predicted Diagnosis',...
    'XDisplayLabels',{'Normal','metastases', 'malignant','fibrosis'},'YDisplayLabels',{'Normal','metastases', 'malignant','fibrosis'},'ColorbarVisible','off');
%The following functions compute the precision and F1 Score
precisionFunc = @(confusionMat) diag(confusionMat)./sum(confusionMat,2);
recallFunc = @(confusionMat) diag(confusionMat)./sum(confusionMat,1);
f1ScoresFunc = @(confusionMat) 2*(precisionFunc(confusionMat).*recallFunc(confusionMat))./(precisionFunc(confusionMat)+recallFunc(confusionMat));
meanF1Func = @(confusionMat) mean(f1ScoresFunc(confusionMat));
precision = precisionFunc(conMat)
recall = recallFunc(conMat)
f1Score = meanF1Func(conMat)