% Copyright 2015 Xilin Shen and Emily Finn 

% This code is released under the terms of the GNU GPL v2. This code
% is not FDA approved for clinical use; it is provided
% freely for research purposes. If using this in a publication
% please reference this properly as: 

% Finn ES, Shen X, Scheinost D, Rosenberg MD, Huang, Chun MM,
% Papademetris X & Constable RT. (2015). Functional connectome
% fingerprinting: Identifying individuals using patterns of brain
% connectivity. Nature Neuroscience 18, 1664-1671.

% This code provides a framework for implementing functional
% connectivity-based behavioral prediction in a leave-one-subject-out
% cross-validation scheme, as described in Finn, Shen et al 2015 (see above
% for full reference). The first input ('all_mats') is a pre-calculated
% MxMxN matrix containing all individual-subject connectivity matrices,
% where M = number of nodes in the chosen brain atlas and N = number of
% subjects. Each element (i,j,k) in these matrices represents the
% correlation between the BOLD timecourses of nodes i and j in subject k
% during a single fMRI session. The second input ('all_behav') is the
% Nx1 vector of scores for the behavior of interest for all subjects.

% As in the reference paper, the predictive power of the model is assessed
% via correlation between predicted and observed scores across all
% subjects. Note that this assumes normal or near-normal distributions for
% both vectors, and does not assess absolute accuracy of predictions (only
% relative accuracy within the sample). It is recommended to explore
% additional/alternative metrics for assessing predictive power, such as
% prediction error sum of squares or prediction r^2.


clear;
clc;

% ------------ INPUTS -------------------

load('OHBM_CPM_test_data', 'test_mat', 'test_score', 'test_cov');
all_mats  = test_mat;
all_behav = test_score;

no_sub = size(all_mats,3);
threshold = 0.01;
fs_option = 2;
% calculate the true prediction correlation
[true_prediction_r_pos, true_prediction_r_neg, true_prediction_r_com] = predict_behavior(all_mats, all_behav, threshold, fs_option, test_cov);

% number of iterations for permutation testing
no_iterations   = 2;
prediction_r    = zeros(no_iterations,3);
prediction_r(1,1) = true_prediction_r_pos;
prediction_r(1,2) = true_prediction_r_neg;
prediction_r(1,3) = true_prediction_r_com;

% create estimate distribution of the test statistic
% via random shuffles of data lables   
for it=2:no_iterations
    fprintf('\n Performing iteration %d out of %d \n', it, no_iterations);
    new_behav        = all_behav(randperm(no_sub));
    [prediction_r(it,1), prediction_r(it,2), prediction_r(it, 3)] = predict_behavior(all_mats, new_behav, threshold, fs_option, test_cov);    
end

sorted_prediction_r_pos = sort(prediction_r(:,1),'descend');
position_pos            = find(sorted_prediction_r_pos==true_prediction_r_pos);
pval_pos                = position_pos(1)/no_iterations;

sorted_prediction_r_neg = sort(prediction_r(:,2),'descend');
position_neg            = find(sorted_prediction_r_neg==true_prediction_r_neg);
pval_neg                = position_neg(1)/no_iterations;


sorted_prediction_r_com = sort(prediction_r(:,3),'descend');
position_com            = find(sorted_prediction_r_com==true_prediction_r_com);
pval_com                = position_com(1)/no_iterations;


