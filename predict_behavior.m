function [r_pos, r_neg, r_com] = predict_behavior(all_mats, all_behav, thresh, fs_option, cov)

% CPM: with leave one out cross validation
% all_mats: N by N by K,N being the number of ROIs, K being the number of
% subjects
% all_behav: K by 1
% threshold: p value threshold for edge selection
% fs_option: feature selection options
% 1: Pearson correlation
% 2: Spearman correlation (rank)
% 3: Robust regression
% 4: Partial correlation with a covariate
% cov: covariance K by L when fs_option =4 to run partial correlation

if( fs_option==4)
    if( nargin<5)
        disp('Need to input the covariate');
        return;
    end
end

no_sub = size(all_mats,3);
no_node = size(all_mats,1);

behav_pred_pos = zeros(no_sub,1);
behav_pred_neg = zeros(no_sub,1);
behav_pred = zeros(no_sub,1);

if( fs_option==1)
    disp('Edge selected based on Pearson correlation');
elseif( fs_option==2)
    disp('Edge selected based on Spearman correlation');
elseif( fs_option==3)
    disp('Edge selected based on robust regression');
    aa = ones(no_node, no_node);
    aa_upp = triu(aa, 1);
    upp_id = find( aa_upp);
elseif( fs_option==4)
    disp('Edge selected based on partial correlation');
end

fprintf('Leave one out cross validation');
for leftout = 1:no_sub;
    
    train_mats = all_mats;
    train_mats(:,:,leftout) = [];
    train_vcts = reshape(train_mats,[],size(train_mats,3));
    
    train_behav = all_behav;
    train_behav(leftout) = [];
    
    
    if( fs_option==1)
        
        % correlate all edges with behavior using Pearson correlation
        [r_mat, p_mat] = corr(train_vcts', train_behav);
        
    elseif( fs_option==2)
        % correlate all edges with behavior using rank correlation
        [r_mat, p_mat] = corr(train_vcts', train_behav, 'type', 'Spearman');
        
    elseif( fs_option==3)
        warning('off')
        % correlate all edges with behavior using robust regression
        edge_no = length(upp_id);
        r_mat = zeros(no_node, no_node);
        p_mat = zeros(no_node, no_node);
        
        for edge_i = 1: edge_no;
            [~, stats] = robustfit(train_vcts(upp_id(edge_i),:)', train_behav);
            cur_t = stats.t(2);
            r_mat(upp_id(edge_i)) = sign(cur_t)*sqrt(cur_t^2/(no_sub-1-2+cur_t^2));
            p_mat(upp_id(edge_i)) = 2*(1-tcdf(abs(cur_t), no_sub-1-2));  %two tailed
        end
        r_mat = r_mat + transpose(r_mat);
        p_mat = p_mat + transpose(p_mat);
    elseif( fs_option==4)
        % correlate all edges with behavior using partial correlation
        cov_train = cov;
        cov_train(leftout,:) = [];
        [r_mat, p_mat] = partialcorr(train_vcts', train_behav, cov_train);
    end
        
    r_mat = reshape(r_mat,no_node,no_node);
    p_mat = reshape(p_mat,no_node,no_node);
    
    % set threshold and define masks 
    pos_mask = zeros(no_node, no_node);
    neg_mask = zeros(no_node, no_node);
    
    
    pos_edge = find( r_mat >0 & p_mat < thresh);
    neg_edge = find( r_mat <0 & p_mat < thresh);
    
    pos_mask(pos_edge) = 1;
    neg_mask(neg_edge) = 1;
    
    
%     %-----------------sigmoidal weighting---------------------------%
%     pos_edges = find(r_mat > 0 );
%     neg_edges = find(r_mat < 0 );
%     
%     % covert p threshold to r threshold
%     T = tinv(thresh/2, no_sub-1-2);
%     R = sqrt(T^2/(no_sub-1-2+T^2));
%     
%     % create a weighted mask using sigmoidal function
%     % weight = 0.5, when correlation = R/3;
%     % weight = 0.88, when correlation = R;
%     pos_mask(pos_edges) = sigmf( r_mat(pos_edges), [3/R, R/3]);
%     neg_mask(neg_edges) = sigmf( r_mat(neg_edges), [-3/R, R/3]);
%     %---------------sigmoidal weighting-----------------------------%
    
    % get sum of all edges in TRAIN subs (divide by 2 to control for the
    % fact that matrices are symmetric)
    
    train_sumpos = zeros(no_sub-1,1);
    train_sumneg = zeros(no_sub-1,1);
    
    for ss = 1:no_sub-1;
        train_sumpos(ss) = sum(sum(train_mats(:,:,ss).*pos_mask))/2;
        train_sumneg(ss) = sum(sum(train_mats(:,:,ss).*neg_mask))/2;
    end
    
    % build model on TRAIN subs
    % combining both postive and negative features
    b = regress(train_behav, [train_sumpos, train_sumneg, ones(no_sub-1,1)]);
    b_pos = regress(train_behav, [train_sumpos, ones(no_sub-1,1)]);
    b_neg = regress(train_behav, [train_sumneg, ones(no_sub-1,1)]);
    
    
    % run model on TEST sub
    
    test_mat = all_mats(:,:,leftout);
    test_sumpos = sum(sum(test_mat.*pos_mask))/2;
    test_sumneg = sum(sum(test_mat.*neg_mask))/2;
    
    behav_pred(leftout) = b(1)*test_sumpos + b(2)*test_sumneg + b(3);
    behav_pred_pos(leftout) = b_pos(1)*test_sumpos+b_pos(2);
    behav_pred_neg(leftout) = b_neg(1)*test_sumneg+b_neg(2);
    
end

[r_pos, p_pos] = corr(behav_pred_pos,all_behav);
[r_neg, p_neg] = corr(behav_pred_neg,all_behav);
[r_com, p_com]=corr(behav_pred, all_behav);

