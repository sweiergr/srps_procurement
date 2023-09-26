%{
    Estimate alpha and revenue distribution parameters based on 
    net cost signals from first-step net estimation and cost distributions
    inferred from net_cost_estimation.m

%}

clear
clc
clf('reset')
format('short');
% Define necessary globals.
global N_obs K theta_debug
% SET PAPER SIZE FOR FIGURES TO AVOID LARGE MARGINS.
figuresize(15,12,'cm')
% Set seed in case any simulation is used.
rng(123456);
% Indicate what to do: Redo graphs? Check distributions for logconcavity?
update_plots = 1;
check_logconcavity = 0;
% Prevent graphs from popping up.
set(gcf,'Visible', 'on'); 


%% Load net auction workspace and gross auction parameters.
% Estimated bid function parameters from gross sample.
load(project_paths('OUT_ANALYSIS','net_cost_estimation'));

% Conditional probabilities of DB being pivotal bidder for
% conditional expected revenue computation is calculated in
% estimate_net_auctions.m and saved to workspace there.
% Just illustrate a few statistics here.
hist(db_prob);
title('Distribution of probability of DB being pivotal bidder');

fprintf('Minimum probability of DB being pivotal: %6.4f\n', min(db_prob));
fprintf('Maximum probability of DB being pivotal: %6.4f\n', max(db_prob));
fprintf('Mean probability of DB being pivotal: %6.4f\n', mean(db_prob));
fprintf('Median probability of DB being pivotal: %6.4f\n', median(db_prob));


%% ESTIMATION OF REVENUE AND ALPHA PARAMETERS. 
% Construct regressor matrix: estimating alpha parameters and revenue parameters. 
% Legend:
% X_orig(:,5): frequency of service
% X_orig(:,3): total zkm 
% SW: Here, we experiment with a couple of different revenue specifications.
X_revenue = [X_orig(:,5), X_orig(:,3), data(:,10)];
X_revenue = [data(:,10)./10, log(data(:,8)*10) ./10, log(data(:,6))./10];
X_revenue = [log(data(:,8)*10)];
X_revenue_old = X_revenue;
mean(X_revenue)
var(X_revenue)
hist(X_revenue)
% New experiments with revenue regressors.
% Column 6: length of track: nkm
% Column 7: total track access charges for full contract
% x Column 8: zkm per year for line
% x Column 9: frequency of service (constant within lines of a contract)
% x Column 10: contract duration in (tens of) years
% x Column 15: frequency of service (varying across lines within contract)
% Column 11: zkm per year for full contract (adde dup over all lines within a contract)
% Column 12: dummy for diesel/non-electrified lines.
% Column 13: dummy for used vehicles permitted


% Exp 1: Spec with total zkm and values logged.
X_revenue = log( 10 .* data(:,8) .* data(:,10));
mean(X_revenue)
var(X_revenue)
hist(X_revenue)
% Exp 2 : Spec with total zkm and no log transformation.
X_revenue = data(:,8) .* data(:,10);
mean(X_revenue)
var(X_revenue)
hist(X_revenue)
% % Exp 3: Spec with frequency of service.
X_revenue = data(:,9) ./ 10;
mean(X_revenue)
var(X_revenue)
hist(X_revenue)
% Exp 4: Spec with frequency of service.
% X_revenue = log(data(:,9));
% mean(X_revenue)
% var(X_revenue)
% hist(X_revenue)
% Compare old and experimental X_revenue.
X_revenue_comp = [X_revenue_old, X_revenue];
hist(X_revenue_comp)
% Double-check how to trunacte RHO signal.
% With the cleaned sample, these cutoffs don't matter much.
negative_RHO_threshold = -25;
positive_RHO_trehshold = 30;
% Experiment with new thresholds to see whether it helps avoiding the zero
RHO(RHO<negative_RHO_threshold) = negative_RHO_threshold;
RHO(RHO>positive_RHO_trehshold) = positive_RHO_trehshold;

% Construct anonymous function for likelihood.
min_neg_log_ll_2 = @(theta) log_ll_net_2step(theta, X_revenue, RHO, db_win, kdens_container, N,db_not_old_input, db_prob);


% Maximize likelihood for second net step using gradient-based fminunc.
% Set starting values, structure of parameters vector
% (1): alpha parameters before exponential transformation, typicallt 2 or 3 parameters.
% (2): revenue variance parameters, typically 2 parameters.
% (3): revenue mean parameters, typically 3 parameters.

% Starting values for nonparametric alpha-specification with linear transformation.
% theta_start =  [0.5;0.45;0.4;0.35; 3.4313; 2.0657; -1.0990];
% Starting values for nonparametric alpha-spec with logistic transformation.
% This seems to have worked well.
theta_start =  [0.05;-0.05;-0.1;-0.1; 4.4313; 4.5657; -1.0990];
AAA = X_revenue(:,1) .* theta_start(end) + theta_start(end-1);
mean(AAA)

% Guess for plausible starting values (obtained with ad-hoc experimentation using ars.m)
theta_start =  [0.1;-0.05;-0.1;-0.01; 4.5; -11; 0.6];
AAAA = X_revenue(:,1) .* theta_start(end) + theta_start(end-1);
mean(AAAA)
theta_debug = theta_start;

% Set optimizer options and call optimizer. 
fminsearch_options = optimset('Display','iter-detailed','TolFun', 1E-4, 'TolX', 1E-8, 'MaxFunEvals', 12000, 'MaxIter', 3000);
fminunc_options = optimset('GradObj', 'off','Display','iter-detailed','TolFun', 1E-2, 'TolX', 1E-4, 'MaxFunEvals', 12000, 'MaxIter', 3000);
tic;
[theta_2_opt, neg_log_ll_2_opt] = fminsearch(min_neg_log_ll_2, theta_start,fminsearch_options);

% Save estimated parameters to file.
save(project_paths('OUT_ANALYSIS','theta_net2'),'theta_2_opt');
net_rev_est_time = toc;

%% Compute X_IE at optimal revenue parameters (used as a measure of the winner's curse).
[test_log_ll,~,X_IE_opt,n_ll_problematic] = min_neg_log_ll_2(theta_2_opt);

% Non-parametric version with logistic transformation.
alpha_I = ...
    exp(theta_2_opt(1)) ./ (1.0 + exp(theta_2_opt(1))) .* (N==2) + ...
    exp(theta_2_opt(2)) ./ (1.0 + exp(theta_2_opt(2))) .* (N==3) + ...
    exp(theta_2_opt(3)) ./ (1.0 + exp(theta_2_opt(3))) .* (N==4) + ...
    exp(theta_2_opt(4)) ./ (1.0 + exp(theta_2_opt(4))) .* (N==5);
% Compute alpha_I as residual.
alpha_E = (1-alpha_I) ./ (N-1);
% Check: all alpha-weights sum to one.
test_sum = alpha_I + (N-1) .* alpha_E;

% Compute sums of all rivals' revenue belief.
% First column: what incumbent believes about entrants' revenue signals.
% Second column: what entrant believes about incumbent's and other entrants' revenue signals.
X_IE_sum = [(N-1) .* alpha_E .* X_IE_opt(:,2), (N-2) .* alpha_E .* X_IE_opt(:,2) + alpha_I .* X_IE_opt(:,1)];
% Compute quotient of incumbent's and entrants' revenue belief.
% What exactly is it that we want to compare here?
X_IE_relative = X_IE_sum(:,2) ./ X_IE_sum(:,1);
hist(X_IE_relative);


% Compute summary of alpha results in table.
N_grid = linspace(2,6,5);
alpha_I_grid = ...
    [exp(theta_2_opt(1)) ./ (1.0 + exp(theta_2_opt(1))) , ...
    exp(theta_2_opt(2)) ./ (1.0 + exp(theta_2_opt(2))) ,  ...
    exp(theta_2_opt(3)) ./ (1.0 + exp(theta_2_opt(3))) ,  ...
    exp(theta_2_opt(4)) ./ (1.0 + exp(theta_2_opt(4))), ...
    exp(theta_2_opt(4)) ./ (1.0 + exp(theta_2_opt(4)))];
% Compute alpha_I as residual.
alpha_E_grid = (1-alpha_I_grid) ./ (N_grid-1);
alpha_grid_total = [ alpha_I_grid; alpha_E_grid];
% Check whether alpha-parameters add up to one:
n_grid_test = linspace(2,6,5);
alpha_test_sum = alpha_I_grid + (N_grid-1) .* alpha_E_grid;
fprintf('Check whether all alpha-parameters for each N sum up to one:\nThis statistic should be one: %6.4f\n', (mean(alpha_test_sum==1 & var(alpha_test_sum)==0)));
alpha_N = [alpha_I, alpha_E, N];

% Compute informatinal advantage of incumbent.
info_advantage = alpha_I_grid ./ alpha_E_grid;
ratio_residual_variance = (N_grid - 2) ./ (N_grid-1) + alpha_I_grid.^2 ./ ((N_grid-1) .* alpha_E_grid.^2);
% Same with std.
ratio_residual_std = sqrt( (N_grid - 2) ./ (N_grid-1) + alpha_I_grid.^2 ./ ((N_grid-1) .* alpha_E_grid.^2));

% Compute standard errors and p-values.
[vcov_rev, std_errors_2, t_stats_2, p_values_2] = stats_2(theta_2_opt, min_neg_log_ll_2, 10^-8, 2);

% Compute standard error of alpha parameters for incumbent and entrant.
alpha_SE = zeros(2,4);
t_alpha_grid = zeros(2,4);
p_alpha_grid = zeros(2,4);
for l=1:size(alpha_SE,2)
    % New version with nonparametric alpha-spec.
    % Compute variance of alpha parameters for various N.
    % Alpha_I statistics.
    alpha_SE(1,l) = sqrt(vcov_rev(l,l) .* (exp(theta_2_opt(l)) ./ (1+exp(theta_2_opt(l)))).^2);
    % Alpha_E statistics.
    alpha_SE(2,l) = sqrt(vcov_rev(l,l) .* (exp(theta_2_opt(l)) ./ (1+exp(theta_2_opt(l)))).^2) ./ l;
    % Testing whether alpha_I is different from 1/N.
    t_alpha_grid(1,l) = (alpha_I_grid(1,l) - (1/(l+1))) ./ alpha_SE(1,l);
    t_alpha_grid(2,l) = ( (1-alpha_I_grid(1,l)) ./(l) - (1/(l+1))) ./ alpha_SE(2,l);
end % end loop over N for alpha-SE computation using Delta-method.
p_alpha_grid = 2.* (1-normcdf(abs(t_alpha_grid)));

alpha_SE
p_alpha_grid

% Write estimation results for net bid functions to file.
disp('Estimation results for net auctions - second step:')
% results_net_2 = [theta_2_out, se_2_out, t_2_out, p_val_2_out];
results_net_2 = [theta_2_opt, std_errors_2, t_stats_2, p_values_2]

col_labels = {'Point estimates', 'Standard errors', 't-statistics', 'p-values'};
row_labels_small = {'$\alpha^I_{2}$','$\alpha^I_{3}$','$\alpha^I_{4}$','$\alpha^I_{5+}$','$\sigma_{r0}$','$\beta_{R0}$', '$\beta_{R1}$'};

% Save estimation results in raw format to be formatted in a nicer table using Python.
dlmwrite(project_paths('OUT_ANALYSIS','er_net2.csv'),results_net_2);
fid = fopen(project_paths('OUT_ANALYSIS','er_net2_legend.csv'), 'w');
fprintf(fid, '%s,', row_labels_small{1:6});
fclose(fid);

%% Construct table for expected revenues and variance.
% Mean and variance of parent normal distribution.
mean_rev_aux = ([ones(T,1), X_revenue] * theta_2_opt(end-size(X_revenue,2):end));
% Variance of revenue signal distribution.
% For specification with constant variance
sigma_rev_aux = sqrt(exp(theta_2_opt(end-2))).*ones(N_obs,1);

% Account for truncation to compute actual mean and variance of positive
% revenue distribution.
rsig_aux = mean_rev_aux ./ sigma_rev_aux;
denom = normcdf(rsig_aux);
mean_revenue = mean_rev_aux + (sigma_rev_aux .* normpdf(rsig_aux) ./ denom) ;
sigma_revenue = sqrt((sigma_rev_aux.^2 .* ( 1 - (rsig_aux .* normpdf(rsig_aux) ./ denom) - (normpdf(rsig_aux).^2 ./ denom.^2) )));

fprintf('Mean and median ticket revenues across net auctions: %6.4f and %6.4f.\n',mean(mean_revenue),median(mean_revenue));
fprintf('Mean and median SD of ticket revenues across net auctions: %6.4f and %6.4f.\n',mean(sigma_revenue),median(sigma_revenue));


% Compute difference of residual variance.
diff_residual_variance =  (alpha_I_grid.^2 - alpha_E_grid.^2) .* mean(sigma_revenue.^2);
% Just for completeness: This statistic is not reported anywhere.
diff_residual_variance_rel =  (alpha_I_grid.^2 - alpha_E_grid.^2) ./ ([1,2,3,4,5] .* alpha_E_grid.^2);
% This is the ratio of the residual variance of entrant over incumbent.
res_var_ratio =  (alpha_I_grid.^2 + [0,1,2,3,4] .* alpha_E_grid.^2) ./ ([1,2,3,4,5] .* alpha_E_grid.^2);
% Compute average of residual variance ratio over all auctions.
res_var_ratio_grid = res_var_ratio(1:end-1)';
res_var_ratio_full = res_var_ratio_grid(N-1);
res_var_ratio_mean = mean(res_var_ratio_full)
res_var_ratio_median = median(res_var_ratio_full)

residual_var_I =  ([1,2,3,4,5] .* alpha_E_grid.^2);
residual_var_E =  ([0,1,2,3,4] .* alpha_E_grid.^2 + alpha_I_grid.^2);
% Compute ratio of residual variance (entrant) over residual variance
% (incumbent).
share_res_var = residual_var_E ./ residual_var_I;
% Compute mean of residual variance ratio over all auctions.
res_var_sample = share_res_var(N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sanity check on revenue numbers.
% Compute ratio of winning bid of revenues and vice versa.
share_bid_rev = bid_win ./ mean_revenue;
% Our target here is roughly 0.6 based on industry sources.
share_rev_bid = mean_revenue ./ bid_win;

share_rev_bid_clean = share_rev_bid;
outlier_threshold = 2;
fprintf('Number of auctions with rev/bid ratio of larger than 2: %d.\n',sum(share_rev_bid>outlier_threshold));
share_rev_bid_clean(share_rev_bid_clean>outlier_threshold) = NaN;

% Compute median rev-bid-share, i.e., how much of winning bid can be covered by
% revenues.
median_share_rev_bid = nanmedian(share_rev_bid);
mean_share_rev_bid = nanmean(share_rev_bid);
median_share_rev_bid_clean = nanmedian(share_rev_bid_clean);
mean_share_rev_bid_clean = nanmean(share_rev_bid_clean);

%% To get better idea, plot histogram of mean revenue as function of winning bid.
subplot(1,1,1)
hist(share_rev_bid,75)
title('Histogram of ratio of mean revenue to winning bid','Interpreter','latex');
xlabel('Ratio mean revenue to winning bid','Interpreter','latex');
ylabel('Frequency','Interpreter','latex')
% xlim([0,50])
filename = project_paths('OUT_FIGURES','hist_ratio_revbid');
saveas(gcf,filename,'pdf');
% Remove outliers, i.e., drop auctions where rev/bid implausibly large (not super relevant after sample is cleaned)
subplot(1,1,1)
hist(share_rev_bid_clean,75)
title('Histogram of ratio of mean revenue to winning bid (no outliers)','Interpreter','latex');
xlabel('Ratio mean revenue to winning bid','Interpreter','latex');
ylabel('Frequency','Interpreter','latex')
% xlim([0,50])
filename = project_paths('OUT_FIGURES','hist_ratio_revbid_clean');
saveas(gcf,filename,'pdf');


% Compute share of mean revenues to mean costs.
share_rev_cost = repmat(mean_revenue,1,2) ./ mean_costs;
share_rev_medcost = repmat(mean_revenue,1,2) ./ median_costs;
% Alternatively, compute mean of rev-cost-share,
% after outliers are kicked out.
share_rev_cost_clean = share_rev_cost;
share_rev_cost_clean(share_rev_cost_clean>10) = NaN;
share_rev_medcost_clean = share_rev_medcost;
share_rev_medcost_clean(share_rev_medcost_clean>10) = NaN;



% Compute median rev-cost-share, i.e., how much of costs can be covered by
% revenues.
%% Cost statistics based on mean costs (could be prone to outliers).
% Unweighted average over incumbent and entrant costs (mean).
median_share_rev_cost_uw = nanmedian(0.5 .* share_rev_cost(:,1) + 0.5 .* share_rev_cost(:,2));
mean_share_rev_cost_uw = nanmean(0.5 .* share_rev_cost(:,1) + 0.5 .* share_rev_cost(:,2));
median_share_rev_cost_clean_uw = nanmedian(0.5 .* share_rev_cost_clean(:,1) + 0.5 .* share_rev_cost_clean(:,2));
mean_share_rev_cost_clean_uw = nanmean(0.5 .* share_rev_cost_clean(:,1) + 0.5 .* share_rev_cost_clean(:,2));
% Weighted (by number of bidders in bidder group) average over incumbent and entrant costs.
median_share_rev_cost_w = nanmedian((share_rev_cost(:,1) + (N-1) .* share_rev_cost(:,2)) ./N);
mean_share_rev_cost_w = nanmean((share_rev_cost(:,1) + (N-1) .* share_rev_cost(:,2)) ./N);
median_share_rev_cost_clean_w = nanmedian((share_rev_cost_clean(:,1) + (N-1) .* share_rev_cost_clean(:,2)) ./N);
mean_share_rev_cost_clean_w = nanmean((share_rev_cost_clean(:,1) + (N-1) .* share_rev_cost_clean(:,2)) ./N);
%% Cost statistics based on median costs (probably less prone to outliers).
% Unweighted average over incumbent and entrant costs (median).
median_share_rev_medcost_uw = nanmedian(0.5 .* share_rev_medcost(:,1) + 0.5 .* share_rev_medcost(:,2));
mean_share_rev_medcost_uw = nanmean(0.5 .* share_rev_medcost(:,1) + 0.5 .* share_rev_medcost(:,2));
median_share_rev_medcost_clean_uw = nanmedian(0.5 .* share_rev_medcost_clean(:,1) + 0.5 .* share_rev_medcost_clean(:,2));
mean_share_rev_medcost_clean_uw = nanmean(0.5 .* share_rev_medcost_clean(:,1) + 0.5 .* share_rev_medcost_clean(:,2));
% Weighted (by number of bidders in bidder group) average over incumbent and entrant costs.
median_share_rev_medcost_w = nanmedian((share_rev_medcost(:,1) + (N-1) .* share_rev_medcost(:,2)) ./N);
mean_share_rev_medcost_w = nanmean((share_rev_medcost(:,1) + (N-1) .* share_rev_medcost(:,2)) ./N);
median_share_rev_medcost_clean_w = nanmedian((share_rev_medcost_clean(:,1) + (N-1) .* share_rev_medcost_clean(:,2)) ./N);
mean_share_rev_medcost_clean_w = nanmean((share_rev_medcost_clean(:,1) + (N-1) .* share_rev_medcost_clean(:,2)) ./N);

%% To get better idea, plot histogram of mean revenue as function of winning bid.
subplot(1,1,1)
hist(share_rev_cost,75)
title('Histogram of ratio of mean revenue to mean cost','Interpreter','latex');
xlabel('Ratio mean revenue to mean cost','Interpreter','latex');
ylabel('Frequency','Interpreter','latex')
legend('Incumbent','Entrant')
% xlim([0,50])
filename = project_paths('OUT_FIGURES','hist_ratio_revcost');
saveas(gcf,filename,'pdf');
% Remove outliers, i.e., drop auctions where rev/bid implausibly large. 
subplot(1,1,1)
hist(share_rev_cost_clean,75)
title('Histogram of ratio of mean revenue to mean cost (no outliers)','Interpreter','latex');
xlabel('Ratio mean revenue to mean cost','Interpreter','latex');
ylabel('Frequency','Interpreter','latex')
legend('Incumbent','Entrant')
% xlim([0,50])
filename = project_paths('OUT_FIGURES','hist_ratio_revcost_clean');
saveas(gcf,filename,'pdf');

% Save expected ticket revenues for net contracts.
E_TR_net = mean_revenue;
% Compare expected difference between revenues and costs to winning bid.
E_CR_diff = mean_costs - repmat(E_TR_net,1,2);
% Compute difference for winner's type.
E_CR_diff_win = E_CR_diff(:,1) .* db_win + E_CR_diff(:,2) .* (1-db_win); 
% Compare winners cost-revenue difference to winning bid.
winner_markup = bid_win ./ E_CR_diff_win;

%% Compute how often winner expects to make a profit from operating line.
net_cost_signal_I = RHO(:,1) .* db_win;
net_cost_signal_E = RHO(:,2) .* (1-db_win);
net_cost_signal_I(net_cost_signal_I==0) = [];
net_cost_signal_E(net_cost_signal_E==0) = [];

% Compute share of lines that winning entrant expects to run profitably.
profitable_share_win_I = sum(net_cost_signal_I<0) ./ length(net_cost_signal_I);
profitable_share_win_E = sum(net_cost_signal_E<0) ./ length(net_cost_signal_E);

% Compare expected revenue to expected costs for each line and bidder type.
profitable_share_I = sum((E_TR_net - mean_costs(:,1))>0) ./ T;
profitable_share_E = sum((E_TR_net - mean_costs(:,2))>0) ./ T;
% Looks like the majority of lines can be operated profitably!

% Write estimation results for net bid functions to file.
disp('Estimation results for alpha-parameters: Summary')
results_alpha_summary = [alpha_I_grid; alpha_E_grid; info_advantage];
col_labels = {'$N=2$', '$N=3$', '$N=4$', '$N=5$','$N=6$'};
row_labels = {'$\alpha^I$','$\alpha^E$','$\frac{\alpha_I}{\alpha_E}$'};
matrix2latex(results_alpha_summary,project_paths('OUT_TABLES','resultsalpha.tex'),'headerRow', col_labels, 'headerColumn', row_labels, 'format', '%6.4f', 'caption', 'Implied asymmetry parameters and informational advantage', 'label', 'tab:estresultsalpha');

% Write log file to diary for saving some of the important statistics.
% Delete existing file if it already exists (to avoid appending).
if(exist(project_paths('OUT_ANALYSIS','netrev_estimation.log')));
    delete(project_paths('OUT_ANALYSIS','netrev_estimation.log'));
end
% Write new log-file.
diary(project_paths('OUT_ANALYSIS','netrev_estimation.log'));
fprintf('This log contains a summary of the results from the net auction revenue estimation.\n')
fprintf(strcat('Estimation ran on: \n ',datestr(now),'\n'));
fprintf(strcat('Estimation ran on: \n ',computer,'\n'));

fprintf('\nSTATISTICS ON NET REVENUE ESTIMATION ITSELF \n');
[theta_start theta_2_opt]
fprintf('Value of negative log likelihood at estimated parameter values: \n %6.4f \n ', neg_log_ll_2_opt);
fprintf('Number of non-defined likelihood points at estimated parameter values: %d \n', n_ll_problematic);
fprintf('Estimation took %6.4f minutes. \n \n\n ', net_rev_est_time./60);


fprintf('\nAll means and medians are computed over the auctions in our net sample.\n\n');

fprintf('\nSTATISTICS ON RATIO REVENUE OVER WINNING BID\n');
fprintf('Rodler & Partner goal for revenue / winning bid: 0.66.\n');
fprintf('Mean ratio of revenues over winning bid: %6.4f \n',mean_share_rev_bid);
fprintf('Mean ratio of revenues over winning bid (no outliers): %6.4f \n',mean_share_rev_bid_clean);
fprintf('Median ratio of revenues over winning bid: %6.4f \n',median_share_rev_bid);
fprintf('Median ratio of revenues over winning bid (no outliers): %6.4f \n',median_share_rev_bid_clean);

fprintf('\nSTATISTICS ON RATIO REVENUE OVER MEAN COST\n');
fprintf('Rodler & Partner goal for revenue / mean costs: 0.4.\n');
fprintf(':::Weighted averages over incumbent and entrants:::\n');
fprintf('Mean ratio of revenues over mean cost: %6.4f \n',mean_share_rev_cost_w);
fprintf('Mean ratio of revenues over mean cost (no outliers): %6.4f \n',mean_share_rev_cost_clean_w);
fprintf('Median ratio of revenues over mean cost: %6.4f \n',median_share_rev_cost_w);
fprintf('Median ratio of revenues over mean cost (no outliers): %6.4f \n',median_share_rev_cost_clean_w);
fprintf(':::Unweighted averages over incumbent and entrants:::\n');
fprintf('Mean ratio of revenues over mean cost: %6.4f \n',mean_share_rev_cost_uw);
fprintf('Mean ratio of revenues over mean cost (no outliers): %6.4f \n',mean_share_rev_cost_clean_uw);
fprintf('Median ratio of revenues over mean cost: %6.4f \n',median_share_rev_cost_uw);
fprintf('Median ratio of revenues over mean cost (no outliers): %6.4f \n',median_share_rev_cost_clean_uw);

fprintf('\nSTATISTICS ON RATIO REVENUE OVER MEDIAN COST\n');
fprintf(':::Weighted averages over incumbent and entrants:::\n');
fprintf('Mean ratio of revenues over median cost: %6.4f \n',mean_share_rev_medcost_w);
fprintf('Mean ratio of revenues over median cost (no outliers): %6.4f \n',mean_share_rev_medcost_clean_w);
fprintf('Median ratio of revenues over median cost: %6.4f \n',median_share_rev_medcost_w);
fprintf('Median ratio of revenues over median cost (no outliers): %6.4f \n',median_share_rev_medcost_clean_w);
fprintf(':::Unweighted averages over incumbent and entrants:::\n');
fprintf('Mean ratio of revenues over median cost: %6.4f \n',mean_share_rev_medcost_uw);
fprintf('Mean ratio of revenues over median cost (no outliers): %6.4f \n',mean_share_rev_medcost_clean_uw);
fprintf('Median ratio of revenues over median cost: %6.4f \n',median_share_rev_medcost_uw);
fprintf('Median ratio of revenues over median cost (no outliers): %6.4f \n',median_share_rev_medcost_clean_uw);

fprintf('\nSTATISTICS ON ABSOLUTE REVENUE LEVELS\n');
fprintf('Mean of expected ticket revenues over auctions (OLD VERSION): 4.9.\n');
fprintf('Mean of expected ticket revenues over auctions: %6.4f.\n',mean(mean_revenue));
fprintf('Median of expected ticket revenues over auctions: %6.4f.\n',median(mean_revenue));

fprintf('\nSTATISTICS ON SHARE OF EX ANTE PROFITABLE LINES WITH ZERO SUBSIDY\n');
fprintf('Share of lines that are expected to be profitable for incumbent: %6.4f \n', profitable_share_I);
fprintf('Share of lines that are expected to be profitable for entrant: %6.4f \n', profitable_share_E);

fprintf('\nSTATISTICS ON DIFFEREN RESIDUAL VARIANCES\n');
fprintf('Average of residual variance ratio (entrant/incumbent): %6.4f \n', res_var_ratio_mean);

% Print most relevant statistics.
fprintf('Complete alpha-grid (first row: incumbent, second row: entrant, in columns: different N from 2 to 5):\n');
alpha_grid_total
res_var_ratio_grid
fprintf('Uncleaned mean and median of expected ticket revenues: %6.4f \t %6.4f.\n',mean(mean_revenue), median(mean_revenue));

%% Compute revenue per zkm
% Total zkm per contract (adjusts units to be compatible with transformations done for estimation).
total_zkm = data(:,8) .* 1E6 .* data(:,10)*10;
% Revenue per zkm.
rev_per_zkm = mean_revenue .* 1E7 ./ total_zkm;
fprintf('Mean and median of revenue per zkm statistics: %6.4f and %6.4f.\n',mean(rev_per_zkm),median(rev_per_zkm));

% Compute expected revenues for gross auction sample.
save(project_paths('OUT_ANALYSIS','postestimation_workspace_net'));
% Turn off log file.
diary OFF
