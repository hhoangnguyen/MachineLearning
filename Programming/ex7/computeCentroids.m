function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

# per: https://www.coursera.org/learn/machine-learning/discussions/weeks/8/threads/CHrCBRe9EeevPg7X4ZMqyg

# We need a K x m matrix
K_by_n_boolean_matrix = zeros(K, m);

# Boolean matrix where 1 if (idx == k), 0 otherwise
for k = 1 : K
  K_by_n_boolean_matrix(k, :) = (idx == k);
end;

# each row is sum of examples in each centroid
centroid_sum = K_by_n_boolean_matrix * X;

# count up number of 1s in each row, this is number of example in each centroid
occurence_vector = sum(K_by_n_boolean_matrix, 2);

# average
centroids = bsxfun(@rdivide, centroid_sum, occurence_vector);

% =============================================================


end

