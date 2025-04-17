clear

% Noise levels
Q = diag([1^2, 1^2, 1^2]);
R = diag([10000, 400]);

% Measurement interval
T = 0.08;

% Total time steps
N = 201;

% Dimension of measurement
nz = 2;

% Amount of data
M_train = 10000;
M_test = 1000;

% Train dataset
X_train = zeros(M_train, N, 3);
Z_train = zeros(M_train, N, nz);

for times = 1:M_train
    times
    % Generate true states
    x_true = zeros(3, N);
    x_true(:, 1) = mvnrnd([1, 1, 1], 100*eye(3))'; % Initial state
    for i = 2 : N
        x_true(:, i) = integ_Lorenz(x_true(:, i-1), [0 T]) + mvnrnd([0; 0; 0], Q)';
    end
    % Generate measurement
    z = zeros(nz, N);
    for i = 1 : N
        z(:, i) = h(x_true(:, i)) + mvnrnd(zeros(nz,1), R)';
    end
    
    X_train(times, :, :) = x_true';
    Z_train(times, :, :) = z';
end

Train_data = Z_train(:, 2:end, :);
Train_gt = X_train;

% Test dataset
X_test = zeros(M_test, N, 3);
Z_test = zeros(M_test, N, nz);

for times = 1:M_test
    times
    % Generate true states
    x_true = zeros(3, N);
    x_true(:, 1) = mvnrnd([1, 1, 1], 100*eye(3))'; % Initial state
    for i = 2 : N
        x_true(:, i) = integ_Lorenz(x_true(:, i-1), [0 T]) + mvnrnd([0; 0; 0], Q)';
    end
    % Generate measurements
    z = zeros(nz, N);
    for i = 1 : N
        z(:, i) = h(x_true(:, i)) + mvnrnd(zeros(nz,1), R)';
    end
    
    X_test(times, :, :) = x_true';
    Z_test(times, :, :) = z';
end

Test_data = Z_test(:, 2:end, :);
Test_gt = X_test;

% Save
Train_data = reshape(Train_data, M_train, size(Train_data,2)*size(Train_data,3));
Train_gt = reshape(Train_gt, M_train, size(Train_gt,2)*size(Train_gt,3));
Test_data = reshape(Test_data, M_test, size(Test_data,2)*size(Test_data,3));
Test_gt = reshape(Test_gt, M_test, size(Test_gt,2)*size(Test_gt,3));
save('./test/Train_data.txt', 'Train_data', '-ascii', '-double')
save('./test/Train_gt.txt', 'Train_gt', '-ascii', '-double')
save('./test/Test_data.txt', 'Test_data', '-ascii', '-double')
save('./test/Test_gt.txt', 'Test_gt', '-ascii', '-double')