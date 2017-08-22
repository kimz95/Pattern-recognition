
%% Initialization
clear ; close all; randn('seed',0); 

%% ==================== Part 1: Loading Data from test.data ===================	
data=load('haberman.csv');
X_data1=data(:,1:end-1);	
y_data=data(:,end);

[N,M]=size(X_data1);

%%normalization

X_data=normalization(X_data1);

%% ==================== Part 2: Random Data Split 75% to 25% 	===============

training_range=floor(75*N/100);

X_train=X_data(1:training_range,:)';
y_train=y_data(1:training_range,:)';	

X_test=X_data(training_range+1:end,:)';
y_test=y_data(training_range+1:end,:)';

%% ==================== Part 3: Estimating Two Probabilities ==================
[N,m]=size(X_train);
m_size=length(find(y_train==1));
b_size=length(find(y_train==2));

P=[m_size b_size]'./N;

% %% ==================== Part 4: Estimating Mean and Variance ==================
malignant_data=X_train(:,find(y_train==1));
[m1_hat, S1_hat]=Gaussian_ML_estimate(malignant_data);
benign_data=X_train(:,find(y_train==2));
[m2_hat, S2_hat]=Gaussian_ML_estimate(benign_data);

S_hat=cat(3,S1_hat,S2_hat);
m_hat=[m1_hat m2_hat];

%% ==================== Part 5: Applying Bayesian Classifier  =================
y_bayesian=bayes_classifier(m_hat,S_hat,P,X_test);

%% ==================== Part 6: Estimating Error of Bayes  ====================

err_bayesian = (1-length(find(y_test==y_bayesian))/length(y_test));

fprintf('Bayesian classifier y_test error is %.3f%% \n',err_bayesian*100);

