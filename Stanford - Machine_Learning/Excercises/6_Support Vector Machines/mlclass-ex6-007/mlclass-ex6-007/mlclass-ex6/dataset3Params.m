function [C, sigma, predictions] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

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

% You need to return the following variables correctly.
tmpC = [0.01 0.03 0.1 0.3 1 3 10 30];
tmpSigma = [0.01 0.03 0.1 0.3 1 3 10 30];

m = size(X, 1);
predictions = zeros(m,1);
error = 0;
min = flintmax;

for i=1:length(tmpC)
   for j=1:length(tmpSigma)
       model= svmTrain(X, y, tmpC(i), @(X, Xval) gaussianKernel(X, Xval, tmpSigma(j)));
       predictions = svmPredict(model,Xval);
       error = mean(double(predictions ~= yval));
       if error < min
            min = error;
            C = tmpC(i);
            sigma = tmpSigma(j);
       end
   end
end


% =========================================================================

end
