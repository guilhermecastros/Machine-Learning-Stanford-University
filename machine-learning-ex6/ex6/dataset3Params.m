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

cVec = [0.01, 0.1, 1, 10, 100 , 1000];
sigmaVec = [0.01, 0.1, 1, 10, 100 , 1000];
bestValue = 1000;

for i=1:length(cVec)
    for j=1:length(sigmaVec)
        model = svmTrain(X, y, cVec(i), @(x1, x2) gaussianKernel(x1, x2, sigmaVec(j)));
        predictions = svmPredict(model, Xval);
        if (bestValue > mean(double(predictions ~= yval)))
            C = cVec(i);
            sigma  = sigmaVec(j);
            bestValue = mean(double(predictions ~= yval));
        end
    end
end

% =========================================================================

end
