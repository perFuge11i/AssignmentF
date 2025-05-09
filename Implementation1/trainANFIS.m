
% generate random input data within ranges
n = 1000;
level = -1.1 + 1.1*2 * rand(n, 1); % range [-1.1, 1.1]
rate  = -0.35 + 0.35*2 * rand(n, 1); % range [-0.35, 0.35]
inputData = [level, rate];

%{
scatter(level, rate, 'b', 'filled')
xlabel('Level'), ylabel('Rate'), title('Random Input Data')
%}

%get FIS and generate true outputs
fis = readfis('tank');
targetOutput = evalfis(fis, inputData);

%{
levelRange = linspace(-1.1, 1.1, 50);
rateRange  = linspace(-0.35, 0.35, 50);

[LevelGrid, RateGrid] = meshgrid(levelRange, rateRange);
inputGrid = [LevelGrid(:), RateGrid(:)];
valveOutput = evalfis(fis, inputGrid);
ValveSurface = reshape(valveOutput, size(LevelGrid));
figure
surf(LevelGrid, RateGrid, ValveSurface)
xlabel('Level'), ylabel('Rate'), zlabel('Valve Output')
title('Input Output mapping surfaceView')
%}

% randomly shuffle data
fullData = [inputData, targetOutput];
rng(0);  % random number seed : for reproducibility
fullData = fullData(randperm(size(fullData, 1)), :);

% split into trainging and testing sets 80/20
n = size(fullData, 1);
nTrain = round(0.8 * n);

trainData = fullData(1:nTrain, :);
testData  = fullData(nTrain+1:end, :);

X_train = trainData(:, 1:2);
Y_train = trainData(:, 3);
X_test  = testData(:, 1:2);
Y_test  = testData(:, 3);


%{
% Plot coverage of both sets
figure
scatter(X_train(:,1), X_train(:,2), 15, 'b', 'filled'); hold on
scatter(X_test(:,1),  X_test(:,2),  15, 'r', 'filled');
xlabel('Level'), ylabel('Rate')
legend('Training', 'Testing')
title('Range Coverage of sets')
%}

% Generate an initail FIS
genOptions = genfisOptions('GridPartition');
genOptions.NumMembershipFunctions = 3;
genOptions.InputMembershipFunctionType = 'gaussmf';
initialFIS = genfis(X_train, Y_train, genOptions);

% Train the ANFIS with 50 epochs
trainOptions = anfisOptions('InitialFIS', initialFIS, ...
                        'EpochNumber', 50, ...
                        'DisplayANFISInformation', 1, ...
                        'DisplayErrorValues', 1, ...
                        'DisplayStepSize', 0);
[trainedModel, trainError] = anfis(trainData, trainOptions);

%{
disp(trainError);
levelRange = linspace(-1.1, 1.1, 50);
rateRange  = linspace(-0.35, 0.35, 50);
[LevelGrid, RateGrid] = meshgrid(levelRange, rateRange);

gridInputs = [LevelGrid(:), RateGrid(:)];
gridOutput = evalfis(trainedModel, gridInputs);

surfaceOutput = reshape(gridOutput, size(LevelGrid));

figure
surf(LevelGrid, RateGrid, surfaceOutput)
xlabel('Level'), ylabel('Rate'), zlabel('Predicted Output')
title('Surface View of NFIS')
%}

% get predicted outputs
Y_pred = evalfis(trainedModel, X_test);

%{
%s sort
[sorted_Y_test, sortIdx] = sort(Y_test);
sorted_Y_pred = Y_pred(sortIdx);    


figure
plot(sorted_Y_test, 'k', 'LineWidth', 1.2); hold on
plot(sorted_Y_pred, 'r--', 'LineWidth', 1.2);
legend('Correct Output', 'Predicted Output')
xlabel('Test Data'), ylabel('Output')
title('Correct vs Predicted Output on Test Data')
%}

% calculate error metrics
errors = Y_test - Y_pred;

MSE  = mean(errors.^2);
RMSE = sqrt(MSE);
meanError = mean(errors);
stdError  = std(errors);

%{
fprintf('Error Metrics on Test Data:\n');
fprintf('  Mean Error        = %.4f\n', meanError);
fprintf('  Standard Deviation= %.4f\n', stdError);
fprintf('  MSE               = %.4f\n', MSE);
fprintf('  RMSE              = %.4f\n', RMSE);
%}

%{
figure
scatter(Y_test, Y_pred, 25, 'filled')
hold on
plot([-1 1], [-1 1], 'r--', 'LineWidth', 1.5)  % reference line (perfect prediction)

xlabel('True Valve Signal (Y\_test)')
ylabel('Predicted Valve Signal (Y\_pred)')
title('Prediction vs True Output on Test Data')
legend('Predicted Outputs', 'Ideal Fit (y = x)', 'Location', 'best')
axis equal
grid on
%}

% generate values with extended ranges
level_extnd = linspace(-1.5, 1.5, 50);
rate_extnd = linspace(-0.5, 0.5, 50);
[LevelExtndGrid, RateExtndGrid] = meshgrid(level_extnd, rate_extnd);


input_extnd = [LevelExtndGrid(:), RateExtndGrid(:)];

% Get results from original FIS and ANFIS
output_fis = evalfis(fis, input_extnd);
output_model = evalfis(trainedModel, input_extnd);


%{
Z_fis   = reshape(output_fis, size(LevelExGrid));
Z_anfis = reshape(output_anfis, size(LevelExGrid));

figure
subplot(1,2,1)
surf(LevelExGrid, RateExGrid, Z_fis)
title('Mamdani FIS Output (Extrapolated Inputs)')
xlabel('Level'), ylabel('Rate'), zlabel('Valve')
axis tight

subplot(1,2,2)
surf(LevelExGrid, RateExGrid, Z_anfis)
title('Neuro-Fuzzy Output (Extrapolated Inputs)')
xlabel('Level'), ylabel('Rate'), zlabel('Valve')
axis tight
%}

% Write the model to a file
writefis(trainedModel, 'anfis_controller.fis');
