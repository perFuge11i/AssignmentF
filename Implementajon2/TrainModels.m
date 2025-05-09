% Read dataset and split into input/output sets
data = readtable('student_habits_performance.csv');

diet_numeric = double(categorical(data.diet_quality));
X_raw = [data.study_hours_per_day, ...
     data.social_media_hours, ...
     data.sleep_hours, ...
     diet_numeric, ...
     data.attendance_percentage];

Y_raw = data.exam_score;

% randomly shuffle and split the set 70/30
n = size(X_raw, 1);
rng(1);  
idx = randperm(n);
nTrain = round(0.7 * n);

% Create train/test sets
X_train = X_raw(idx(1:nTrain), :);
Y_train = Y_raw(idx(1:nTrain));

X_test  = X_raw(idx(nTrain+1:end), :);
Y_test  = Y_raw(idx(nTrain+1:end));

%{
figure
scatter(X_train(:,3), X_train(:,5), 40, 'b', 'filled'); hold on
scatter(X_test(:,3),  X_test(:,5),  40, 'r', 'filled');

xlabel('Sleep Hours')
ylabel('Attendance %')
legend('Training Data', 'Test Data')
title('Train/Test Split: 3rd and 5th feature')
grid on
%}

% Generate FIS using subtractive clustering
genOpt = genfisOptions('SubtractiveClustering');
genOpt.ClusterInfluenceRange = 0.9;  % Few rules to reduce number of variables (1460)
initialFIS = genfis(X_train, Y_train, genOpt);

% Get tunable settings
tuneSettings = getTunableSettings(initialFIS);

%%%%%%%%%%%%%%%%%%%%%
% Chat-GPT suggestion to remove total variables

for i = 1:numel(tuneSettings)
    for j = 1:numel(tuneSettings(i).MembershipFunctions)
        tuneSettings(i).MembershipFunctions(j) = setTunable( ...
            tuneSettings(i).MembershipFunctions(j), true);
    end
end

%%%%%%%%%%%%%%%%%%%%%

trainData = [X_train, Y_train];


% GA

% options
gaOpt = tunefisOptions('Method', 'ga', ...
                        'OptimizationType', 'learning', ...
                        'Display', 'all');
% Reduced ga parameters for faster computation time
gaOpt.MethodOptions.MaxGenerations = 50;    
gaOpt.MethodOptions.PopulationSize = 30;

% training
trainedFIS_ga = tunefis(initialFIS, tuneSettings, X_train, Y_train, gaOpt);

% PSO

% options
psoOpt = tunefisOptions('Method', 'particleswarm', ...
                         'OptimizationType', 'learning', ...
                         'Display', 'all');
% Reduced pso parameters for faster computation time
psoOpt.MethodOptions.MaxIterations = 20;
psoOpt.MethodOptions.SwarmSize = 30;

% training
trainedFIS_pso = tunefis(initialFIS, tuneSettings, X_train, Y_train, psoOpt);


% ANFIS

% options -- 100 epochs
anfisOpt = anfisOptions('InitialFIS', initialFIS, ...
                        'EpochNumber', 100, ...
                        'DisplayANFISInformation', 0, ...
                        'DisplayErrorValues', 1);

% training
[trainedFIS_anfis, trainError] = anfis(trainData, anfisOpt);


% Evaluate models
Y_pred_ga    = evalfis(trainedFIS_ga, X_test);
Y_pred_pso   = evalfis(trainedFIS_pso, X_test);
Y_pred_anfis = evalfis(trainedFIS_anfis, X_test);

% Compute Error for each
rmse_ga    = sqrt(mean((Y_test - Y_pred_ga).^2));
rmse_pso   = sqrt(mean((Y_test - Y_pred_pso).^2));
rmse_anfis = sqrt(mean((Y_test - Y_pred_anfis).^2));

%{
fprintf('Test RMSE:\n');
fprintf('  GA:    %.4f\n', rmse_test_ga);
fprintf('  PSO:   %.4f\n', rmse_test_pso);
fprintf('  ANFIS: %.4f\n', rmse_test_anfis);
%}

%{
% sorting
[sorted_Y_test, sortIdx] = sort(Y_test, 'descend');

sorted_ga    = Y_pred_ga(sortIdx);
sorted_pso   = Y_pred_pso(sortIdx);
sorted_anfis = Y_pred_anfis(sortIdx);

% Plot GA
figure
plot(sorted_Y_test, 'k', 'LineWidth', 1.5); hold on
plot(sorted_ga, 'b-', 'LineWidth', 1.2)
legend('True Output', 'GA Prediction')
xlabel('Sorted Sample Index (High to Low)')
ylabel('Predicted Output')
title('GA vs True Output (Test Set)')
grid on

% Plot PSO
figure
plot(sorted_Y_test, 'k', 'LineWidth', 1.5); hold on
plot(sorted_pso, 'g-', 'LineWidth', 1.2)
legend('True Output', 'PSO Prediction')
xlabel('Sorted Sample Index (High to Low)')
ylabel('Predicted Output')
title('PSO vs True Output (Test Set)')
grid on

% Plot ANFIS
figure
plot(sorted_Y_test, 'k', 'LineWidth', 1.5); hold on
plot(sorted_anfis, 'r-', 'LineWidth', 1.2)
legend('True Output', 'ANFIS Prediction')
xlabel('Sorted Sample Index (High to Low)')
ylabel('Predicted Output')
title('ANFIS vs True Output (Test Set)')
grid on
%}