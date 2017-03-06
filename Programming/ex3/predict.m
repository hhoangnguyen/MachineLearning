function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add ones to the X data matrix, 16x3
X = [ones(m, 1) X];

% Layer 2
z_sup_2 = X * Theta1'; % 16x3 * 3*4 = 16x4
a_sup_2 = sigmoid(z_sup_2); % 16x4

% Add ones to the a_sup_2 data matrix 16x5
a_sup_2 = [ones(m, 1) a_sup_2];

% Layer 3
z_sup_3 = a_sup_2 * Theta2'; % 16x5 * 5x4 = 16x4
a_sup_3 = sigmoid(z_sup_3); % 16x4

% Make prediction
[val idx] = max(a_sup_3, [], 2);
p = idx;

% =========================================================================


end
