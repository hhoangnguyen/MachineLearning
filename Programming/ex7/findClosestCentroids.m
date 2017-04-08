function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

for example_index = 1 : size(X,1)
  % get each row = example
  current_example = X(example_index, :);
  
  % subtract a 15x11 matrix from 1x11 matrix row by row
  difference = bsxfun(@minus, centroids, current_example);
  
  % get the square norm, square root doesn't really make a difference
  square_norm = sum(difference.^2, 2);
  
  % index of mininum square_norm is index of closest centroid
  [val, index] = min(square_norm);
  idx(example_index) = index;

% =============================================================

end

