%% Instructions:
% This code is a template, it is just like the code for cross validation 
% you used in previous assignments, it should be very familiar to you by now!
% You should take this code and incorporate it into assignment5.m, adapting 
% it as needed.
clear all;clc;close all;

doSoftMax     = false;
doReLU        = true;

[features,labels,posterior] = construct_data(300,'train','nonlinear');
%% drop the constant term
X = features([1,2],:)';
%% labels change from 0,1 to 1,2 (required by the code below)
y = labels' + 1;
m = size(X, 1); % useful variable 
nsamples = size(X,1);

N_range      = [5 10 20 50 100]; % range of nb of layer in the hidden layer
N_lambdas    = 5; % number of lambdas that we test
lambda_range = linspace(0.00001,0.0005,N_lambdas);

K            = 10; % number of cross validation folds

cv_error = zeros(size(lambda_range,2),size(N_range,2));  % initialise the matrix to store cross val errors

for i=1:size(lambda_range,2) % iterate over lambdas
    lambda = lambda_range(1,i);
    for j=1:size(N_range,2) % iterate over number of neurons in hidden layer
        N      = N_range(1,j);
        %% Specify network architecture
        %% format: input dimension, # hidden notes at different layers, output dimension
        nnodes =[2,N,2]; % neural net architecture
        error  = zeros(1,K);
        
        for k=1:K
            randn('seed',0); % (but make it repeatable)
            
            nHidden    = length(nnodes)-1;
            initial_value = [];
            for l=1:nHidden,
                %% add one for the constant component
                n_inputs        = nnodes(l) + 1;
                %% target neuronsn_
                n_outputs         = nnodes(l+1);
    
                %% standard deviation of Gaussian distribution used for initialization
                sigma           = .1;
                WeightsLayer    = randn(n_inputs,n_outputs)*sigma;
    
                %% collate everything in one big parameter vector
                initial_value  = [initial_value;WeightsLayer(:)];
            end
            
            %% TEMPLATE FOR CROSS-VALIDATION CODE 
            %split data into training set and validation set
            [training_set_inputs,training_set_targets,validation_set_features,...
                validation_set_targets] =  split_data(X',labels,nsamples,K,k); % other wise labels

            %Training the neural network        
            
            %% optimizer options
            options = optimset('MaxIter', 500);

            %% our optimization function (mincg) requires creating a pointer to
            %% the function that is being minimized
            costFunction = @(p) nnet(p, ...
            nnodes,training_set_inputs',(training_set_targets'+1), lambda,...
            doSoftMax,doReLU);

            [nn_params, cost] = fmincg(costFunction, initial_value, options);  % initialise vec of params  
            
            %Estimate the error for each validation_run          
            pred = nnet(nn_params,nnodes, validation_set_features',[],[],doSoftMax,doReLU); % get posteriors
            [not_needed, predicted_label] = max(pred,[],2); % convert into prediction
            error(1,k) = 1-sum(predicted_label == (validation_set_targets'+1))/size(predicted_label,1); % compute the prediction error                         
        end
        %The generalization error is the mean of the error
        cv_error(i,j)=mean(error,2); % store error in the cv_error matric
    end
end
colormap hsv;
figure,imagesc(cv_error)


%Pick the best lambda and N - those that minimize the cv_error





