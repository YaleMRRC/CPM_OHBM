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
no_fold = 10;
no_iter = 10;


fs_option = 4;
if( fs_option==1)
     [all_r, all_p, pos_mat, neg_mat, pred_all] =  kfold_cpm_withmatrix(all_mats, all_behav, no_fold, no_iter, threshold, 'Pearson');
elseif( fs_option==2)
    [all_r, all_p, pos_mat, neg_mat, pred_all] =  kfold_cpm_withmatrix(all_mats, all_behav, no_fold, no_iter, threshold, 'Spearman');
elseif( fs_option==4)
    [all_r, all_p, pos_mat, neg_mat, pred_all] =  kfold_cpm_partial_withmatrix(all_mats, all_behav, test_cov, no_fold, no_iter, threshold, 'Pearson');
end

disp(['Average R over ' num2str(no_iter) ' iterations of ', num2str(no_fold), '-fold cross validation is ', num2str(mean(all_r))]);
figure; boxplot(all_r);
output_flg = 1;

if( output_flg==1)
    [sort_v, sort_id ] = sort(all_r);
    half_fold = round(no_fold/2);
    out_pos = pos_mat{ sort_id(half_fold)};
    out_neg = neg_mat{ sort_id(half_fold)};
    % save the positive edge mask and the negative edge mask
    dlmwrite('cpm_kfold_test_pos_mask', out_pos, 'delimiter', '\t');
    dlmwrite('cpm_kfold_test_neg_mask', out_neg, 'delimiter', '\t');
end
