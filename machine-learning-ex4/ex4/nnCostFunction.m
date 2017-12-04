function [J grad] = nnCostFunction(nn_params, ...  %vector with unrolled thetas (incl bias)
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

X = [ones(m,1) X];

a2 = [ones(m,1) sigmoid(X * Theta1')]; % m x hidden_layer_size + bias (26)

h = sigmoid(a2 * Theta2'); % m x num_lables

Y = y == [1:num_labels]; % m x num_lables

J = (1 / m) * sum(sum(-Y .* log(h) - (1 - Y) .* log(1 - h))) + ...
lambda * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))) / (2*m);

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

%for i = 1:m

    % PART 2.1

    %a1 = X(i,:)'; % input size column vector (401)
    %z2 = Theta1 * a1; % hidden layer size column vector (25)
    %a2 = [1; sigmoid(z2)]; % hidden layer size column vector + bias(26)
    %z3 = Theta2 * a2; % output layer size(10)
    %a3 = sigmoid(z3); % output layer size(10)

    %vectorized
              z2 = X * Theta1'; % mx401 * 401x25 = mx25
              a2 = [ones(m,1) sigmoid(z2)]; % mx26
              z3 = a2 * Theta2'; % mx26 * 26x10 = mx10
              a3 = sigmoid(z3); % mx10

    % PART 2.2
    %d3 = a3 - Y(i,:)'; %10x1

    %vectorized
              d3 = a3 - Y; %mx10 => each row is d3 for each sample

    % PART 2.3
    % 25x10 * 10x1 .* 25x1
    %d2 = Theta2(:,2:end)' * d3 .* sigmoidGradient(z2);

    %vectorized  mx10 * 10x25 = mx25 .* mx25
                  %each row is d2 for each sample
              d2 = d3 * Theta2(:,2:end) .* sigmoidGradient(z2);

    % PART 2.4
    % 25x401 = 25x401 + ( 25x1 * 1x401 )
    %Theta1_grad = Theta1_grad + d2 * a1';

    %10x26 + ( 10x1 * 1x26 )
    %Theta2_grad = Theta2_grad + d3 * a2';

    %Vectorized
              Theta1_grad = d2' * X;
              Theta2_grad = d3' * a2;

%  end

Theta1_grad = Theta1_grad / m;

Theta2_grad = Theta2_grad / m;



% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda * Theta1(:,2:end) / m;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda * Theta2(:,2:end) / m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
