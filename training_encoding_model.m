% This script is used to build the linear encoding model between word2vec
% features and MRI responses on natural speech stimuli (collected from
% Feburary 2018 to July 2018).
%% Define data directory
root_dir = 'Home_Directory/data/';
code_dir = 'Home_Directory/code/';
addpath(genpath(code_dir))
%% Training encoding model with ridge regression
load([root_dir, 'encoding_dataset/Concatenated_train.mat'])

% Standardize the predictors (word2vec features) for linear regression
% m = mean(word2vec_train);
% s = std(word2vec_train);
% save([root_dir, 'encoding_dataset/training_statistics.mat'], 'm' ,'s')

% Standardize the predictors (word2vec features) for linear regression
load([root_dir, 'encoding_dataset/training_statistics.mat'])
word2vec_train_std = word2vec_train;
for i = 1:300
    word2vec_train_std(:,i) = (word2vec_train(:,i) - m(i))./s(i);
end

% Optional: change data to single precision
data_train = single(data_train);
word2vec_train_std = single(word2vec_train_std);

% create a list of potential regularization parameters
k = 0:0.5:5;
lambda = 10.^(k);
reguList = lambda;
crossNum = 10;

% training
[thetaFinal, validSquaError, CV, reguFinal, thetaRecord] = encoding(data_train', word2vec_train_std, reguList, crossNum);
save([root_dir,'encoding_result/encoding_training_result.mat'], 'thetaRecord', 'thetaFinal', 'validSquaError', 'CV', 'reguFinal', '-v7.3')
%% Cross-validating encoding model
% set parameters
trial_num = 100000;
window = 30;
cross_num = 10;
operation_effi = true;
% optional: using gpu to accelerate
use_gpu = true;
gpu_device = 1;
if use_gpu
    input = gpuArray(word2vec_train_std);
    fmri_signal = gpuArray(data_train');
end

% run permutation test
[avg_r, count, ~] = validation_permute(input, fmri_signal, trial_num, window, ...
    'regularization', 10.0, 'use_gpu', use_gpu, 'gpu_device', gpu_device, 'cross_num', cross_num, 'operation_effi', operation_effi);

% statistics
permu_record = gather(count);
trial_sum_all = gather(trial_num);
P = (permu_record+1)/(trial_sum_all+1);
% multiple correction
[h, ~, ~, adj_p]=fdr_bh(P,0.05,'dep','yes');
mask = (adj_p<0.05)'; % a map of "semantic system" that contains significantly predictable voxels
save([root_dir,'encoding_result/cross_validation_permutation_result.mat'], 'adj_p', 'mask', 'trial_sum_all', '-v7.3')
%% Testing encoding model
load([root_dir, 'encoding_dataset/Concatenated_test.mat'])
load([root_dir, 'encoding_dataset/training_statistics.mat'])

word2vec_test_std = word2vec_test;
for i = 1:300
    word2vec_test_std(:,i) = (word2vec_test(:,i) - m(i))./s(i);
end
X_pred = word2vec_test_std*thetaFinal;

% Optional: offset first few time points
offset = 4;
data_test = data_test(:,offset:end);
X_pred = X_pred(offset:end,:);

% temporally smooth data_test
data_test_sm = data_test;
for i = 1:size(data_test,1)
    data_test_sm(i,:) = smooth(data_test(i,:));
end

% calculate Pearson correlation between true and predicted testing data
R = correffi(data_test_sm',X_pred);
save([root_dir,'encoding_result/encoding_testing_result'], 'X_pred', 'data_test', 'data_test_sm', 'R', 'offset', '-v7.3')
%% Permutation test for encoding performance on testing data
% test_fmri:  voxel-wise time series for testing data (size: 1 x time_length x voxel_size)
% Note that testing data temporal length after offset 4 is 540
test_fmri = zeros(1, 540, 59412);
test_fmri(1,:,:) = data_test_sm';
% test_weight: encoding parameters (embedding_size x voxel_size)
test_weight = thetaFinal;
% test_conca_sig: word2vec time series for testing data (size: 1 x time_length x embedding_size)
test_concat_sig = zeros(1, 540, 300);
test_concat_sig(1,:,:) = word2vec_test_std;

% permutation test
window = 30;
trial_num = 100000;
[permu_record, ~] = test_permute(test_weight, test_fmri, test_concat_sig, 'window',window,'trial_num',trial_num,'usegpu',1);
save([root_dir,'encoding_result/testing_permutation_result'], 'window', 'trial_num', 'permu_record')

% multiple correction
P = (permu_record+1)/(trial_num+1);
[h, ~, ~, adj_p]=fdr_bh(P,0.05,'dep','yes');