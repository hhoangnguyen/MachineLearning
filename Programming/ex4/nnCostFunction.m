function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% h_theta_x
a_sup_1 = [ones(m, 1) X];

z_sup_2 = a_sup_1 * Theta1';
a_sup_2 = sigmoid(z_sup_2);
a_sup_2 = [ones(m, 1) a_sup_2];

z_sup_3 = a_sup_2 * Theta2';
h_theta_x = sigmoid(z_sup_3);

% Since each y is a scalar, we need to convert it to a 1x10 vector where num_labels = 10
% We do this by create a mx10 zeros matrix, looping through each example, 
% then assign 1 to appropriate column of new_y
new_y = zeros(m, num_labels);
for currentSample = 1 : m
  new_y(currentSample, y(currentSample)) = 1;
end

% Vectorized
cost = new_y' * log(h_theta_x) + (1 .- new_y)' * log(1 .- h_theta_x);

% Sum up main diagonal because that's all that we need.
% Reference: https://www.coursera.org/learn/machine-learning/discussions/weeks/5/threads/ag_zHUGDEeaXnBKVQldqyw
new_cost = 0;
for i = 1 : num_labels  
  new_cost = new_cost + cost(i, i);
end

% Unregularized cost function
J = -(1/m) * new_cost;

% Note: element wise operation also works.
% This is somewhat cleaner because we don't have to sum up diagonals
% sum(sum(new_y .* log(h_theta_x) + (1 .- new_y) .* log(1 .- h_theta_x)))

% we don't regularize bias units
newTheta1 = Theta1;
newTheta1(:,1) = 0;
newTheta2 = Theta2;
newTheta2(:,1) = 0;
regularized_cost_param = lambda/(2*m) * (sum(sum(power(newTheta1, 2))) + sum(sum(power(newTheta2, 2))));

% Regularize cost function
J = J + regularized_cost_param;

% -------------------------------------------------------------
% Backpropagation algorithm, loop version
for l = 1 : m
  % Step 1: Feedforward, get output
  a_sup_1 = [1 X(l, :)];
  z_sup_2 = a_sup_1 * Theta1';
  a_sup_2 = sigmoid(z_sup_2);
  a_sup_2 = [1 a_sup_2];
  z_sup_3 = a_sup_2 * Theta2';
  
  h_theta_x = sigmoid(z_sup_3);
  
  % Step 2: Error in output layer
  yVec = (1:num_labels) == y(l);
  delta_3 = h_theta_x - yVec;
  
  % Step 3: Error in hidden layer 2  
  % don't include first column of Theta2  
  delta_2 = (delta_3 * Theta2(:, 2:end)) .* sigmoidGradient(z_sup_2);
  
  Theta2_grad = Theta2_grad + delta_3' * a_sup_2;
  Theta1_grad = Theta1_grad + delta_2' * a_sup_1;
    
end

Theta2_grad = Theta2_grad/m;
Theta1_grad = Theta1_grad /m;

% regularization
Theta1_grad = Theta1_grad + (lambda / m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = Theta2_grad + (lambda / m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
