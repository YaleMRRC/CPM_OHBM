function [all_r, all_p, pos_mat, neg_mat, pred_all] =  kfold_cpm_withmatrix(mat, score, no_fold, no_iter, thresh, type)


% mat is N by N by #subjects
% score is the behavivoral score is N by 1
% no_fold number of folds
% no_iter number of iteration
% thresh P threshold
if(nargin<6)
    type = 'Pearson';
end

no_nodes = size( mat, 1);
no_sub = size(mat, 3);
mat = reshape(mat, no_nodes*no_nodes, no_sub);

aa = ones( no_nodes, no_nodes);
aa_upp = triu(aa,1);
upp_id = find( aa_upp>0);


per_fold = round( no_sub/no_fold);

all_r = zeros(no_iter, 1);
all_p = zeros(no_iter, 1);
pos_mat = cell(1, no_iter);
neg_mat = cell(1, no_iter);
pred_all = zeros(no_sub, no_iter);


for iter = 1: no_iter;
    
    cur_order = randperm(no_sub);
    
    pred = zeros(no_sub, 1);
    
    for fold = 1: no_fold
        cur_fold = cur_order((fold-1)*per_fold+1: min( fold*per_fold, no_sub));
        % cur_fold is the set of subjects to left out
        cur_no_train_sub = no_sub-length(cur_fold);
        cur_no_test_sub = length(cur_fold);
           
        train_data = mat(upp_id,:);
        train_data(:, cur_fold) = [];
        train_score = score;
        train_score(cur_fold)= [];
            
        test_data = mat(upp_id, cur_fold);
        test_score = score(cur_fold);
        
        % correlate all edges with behavior using Pearson correlation
        [r_mat, p_mat] = corr(train_data', train_score, 'type', type);
        
        
        pos_edge = find( r_mat >0 & p_mat < thresh);
        neg_edge = find( r_mat <0 & p_mat < thresh);
        
        if( fold ==1)
            cc_pos = zeros(no_nodes, no_nodes);
            cc_pos(upp_id(pos_edge)) = 1;
        
            cc_neg = zeros(no_nodes, no_nodes);
            cc_neg(upp_id(neg_edge)) = 1;
        else
            cc_pos(upp_id(pos_edge)) = cc_pos(upp_id(pos_edge))+1;
            cc_neg(upp_id(neg_edge)) = cc_neg(upp_id(neg_edge))+1;
        end
        
        
        train_sumpos = sum(train_data(pos_edge, :));        
        train_sumneg = sum(train_data(neg_edge, :));
        
        test_sumpos = sum(test_data(pos_edge,:));
        test_sumneg = sum(test_data(neg_edge,:));
        
        b = regress(train_score, [train_sumpos', train_sumneg', ones(cur_no_train_sub,1)]);
        test_estimate = [test_sumpos', test_sumneg', ones(cur_no_test_sub,1)]*b;
        
        pred(cur_fold) = test_estimate;
    end
        
    pred_all(:, iter) = pred;
    
    [r, p] = corr( score, pred);
    all_r(iter) = r;
    all_p(iter) = p;
    
    apos = (cc_pos>(fold/2));
    pos_mat{iter} = apos+transpose(apos);
    aneg = (cc_neg>(fold/2));
    neg_mat{iter} = aneg+transpose(aneg);
end

% % define your saving location: save_path
% dlmwrite([save_path, 'cpm_results_r'], all_r, 'delimiter', '\t');
% dlmwrite([save_path, 'cpm_results_p'], all_p, 'delimiter', '\t');