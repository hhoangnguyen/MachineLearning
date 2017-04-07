function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% start with a very large error
min_prediction_error = 10000000; # is there an Infinity constant in Octave?

# try different values of C and sigma per suggestion
C_array = [0.03 0.01 0.3 0.1 1 3 10 30]';
sigma_array = C_array;

# loop through each C
for C_index = 1 : size(C_array)
  # loop through each sigma
  for sigma_index = 1 : size(sigma_array)
    # save current value
    C_current = C_array(C_index);
    sigma_current = sigma_array(sigma_index);  
    
    # calculate model using svmTrain
    model = svmTrain(X, y, C_current, @(x1, x2) gaussianKernel(x1, x2, sigma_current));
    
    # make prediction using svmPredict
    prediction = svmPredict(model, Xval);
    
    # calculate prediction error
    prediction_error = mean(double(prediction ~= yval));
    
    # if we find smaller error, update min_prediction_error, C, and sigma
    if (prediction_error < min_prediction_error)
      min_prediction_error = prediction_error;
      C = C_current;
      sigma = sigma_current;
    end;
  end;
end;

% =========================================================================

end
