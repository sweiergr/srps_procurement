%{
    "Master file" to call various counterfactual computations, i.p., expected subsidies and efficiency
    probabilities for different procurement formats. The file calls separate fies for each
    auction mode.

%}

clear
clc
format('short');
% Define necessary globals.
global N_obs K
% Incumbent in first column, entrant in second column.
clf('reset')
% SET PAPER SIZE FOR FIGURES TO AVOID LARGE MARGINS.
figuresize(15,12,'cm')

%% LOAD RELEVANT WORKSPACES FROM MAT-FILES.
load(project_paths('OUT_ANALYSIS','cf_prep_gross_auctions'));
load(project_paths('OUT_ANALYSIS','cf_prep_net_auctions'));
load(project_paths('OUT_ANALYSIS','cf_prep_cfnetgross_auctions'));

%% Print comparison of efficiency and winning bid statistics for different ways of accounting for no entrant entry probability.
fid = fopen(project_paths('OUT_ANALYSIS','cf_compare_stats.log'),'wt');
fprintf(fid,'Comparison of effect of allowing for no entrant entering versus conditioning on at least one entrant entering\n');

fprintf(fid,'Gross auctions: Mean and median of revenue per zkm statistics: %6.4f and %6.4f.\n',mean(rev_per_zkm),median(rev_per_zkm));
fprintf(fid,'Gross auctions: Expected revenue (mean and median across gross auctions): %6.4f and %6.4f.\n',mean(mean_revenue),median(mean_revenue));

fprintf(fid,'Mean and median efficiency probabilities in gross auctions:\n %6.4f \t %6.4f \t for constant number of bidders\n%6.4f \t %6.4f \t for endogenous number of bidders \n%6.4f \t %6.4f \t for endogenous number of bidders allowing for no entrant entering\n',nanmean(Pr_efficient_gross_constant_n),nanmedian(Pr_efficient_gross_constant_n), nanmean(Pr_efficient_gross),nanmedian(Pr_efficient_gross), nanmean(Pr_efficient_gross_backup),nanmedian(Pr_efficient_gross_alt));
fprintf(fid,'Mean and median expected winning bid in gross auctions:\n %6.4f \t %6.4f \t for constant number of bidders\n%6.4f \t %6.4f \t for endogenous number of bidders\n%6.4f \t %6.4f \t for endogenous number of bidders allowing for no entrant entering',nanmean(E_subsidy_gross_constant_n),nanmedian(E_subsidy_gross_constant_n), nanmean(E_subsidy_gross),nanmedian(E_subsidy_gross), nanmean(E_subsidy_gross_backup),nanmedian(E_subsidy_gross_alt));
fprintf(fid,'Mean and median efficiency probabilities in counterfactual Net-> Gross:\n %6.4f \t %6.4f \t for constant number of bidders\n%6.4f \t %6.4f \t for endogenous number of bidders\n %6.4f \t %6.4f \t for endogenous number of bidders allowing no entrant entering\n',nanmean(Pr_efficient_cfnetgross_constant_n),nanmedian(Pr_efficient_cfnetgross_constant_n), nanmean(Pr_efficient_cfnetgross),nanmedian(Pr_efficient_cfnetgross), nanmean(Pr_efficient_cfnetgross_backup),nanmedian(Pr_efficient_cfnetgross_alt));
fprintf(fid,'Mean and median expected winning bid in counterfactual Net-> Gross:\n %6.4f \t %6.4f \t for constant number of bidders\n%6.4f \t %6.4f \t for endogenous number of bidders \n%6.4f \t %6.4f \t for endogenous number of bidders allowing for no entrant entering\n',nanmean(E_subsidy_cfnetgross_constant_n),nanmedian(E_subsidy_cfnetgross_constant_n), nanmean(E_subsidy_cfnetgross),nanmedian(E_subsidy_cfnetgross), nanmean(E_subsidy_cfnetgross_backup),nanmedian(E_subsidy_cfnetgross_alt));
fprintf(fid,'Mean and median efficiency probabilities in net auctions:\n %6.4f \t %6.4f \t for constant number of bidders\n%6.4f \t %6.4f \t for endogenous number of bidders \n%6.4f \t %6.4f \t for endogenous number of bidders allowing for no entrant entering\n',nanmean(Pr_efficient_net_constant_n),nanmedian(Pr_efficient_net_constant_n), nanmean(Pr_efficient_net),nanmedian(Pr_efficient_net_backup), nanmean(Pr_efficient_net_alt),nanmedian(Pr_efficient_net_alt));
fprintf(fid,'Mean and median expected winning bid in net auctions:\n %6.4f \t %6.4f \t for constant number of bidders\n%6.4f \t %6.4f \t for endogenous number of bidders\n%6.4f \t %6.4f \t for endogenous number of bidders allowing for no entrant entering',nanmean(E_subsidy_net_constant_n),nanmedian(E_subsidy_net_constant_n), nanmean(E_subsidy_net),nanmedian(E_subsidy_net_backup), nanmean(E_subsidy_net_alt),nanmedian(E_subsidy_net_alt));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% START OF FORMATTING RESULTS FOR ENDOGENOUS N
disp('---ANALYZE RESULTS FOR ENDOGENOUS NUMBER OF BIDDERS---');
% Compute relative efficiency changes when going from gross to net.
rel_eff_change_netgross = (Pr_efficient_cfnetgross - Pr_efficient_net) ./ Pr_efficient_net;
% Count efficiency gains and losses.
eff_change_netgross  = rel_eff_change_netgross .* Pr_efficient_net;
sum_eff_gain = sum(rel_eff_change_netgross>=0) ./ length(Pr_efficient_cfnetgross)
sum_eff_loss = sum(rel_eff_change_netgross<0) ./ length(Pr_efficient_cfnetgross)

% Compute mean efficiency gain.
mean_eff_loss = mean(eff_change_netgross(eff_change_netgross<0))
mean_eff_gain = mean(eff_change_netgross(eff_change_netgross>0 & eff_change_netgross<=1))
sprintf('Share of net tracks for which gross leads to efficiency gain: %.2f \n', sum_eff_gain)
sprintf('Share of net tracks for which gross leads to efficiency loss: %.2f \n', sum_eff_loss)
sprintf('Average efficiency gain when net tracks procured as gross: %.2f \n', mean_eff_gain)
sprintf('Average efficiency loss when net tracks procured as gross: %.2f \n', mean_eff_loss)


%% Compare revenues and agency payoffs first.
% Check if some lines have very bad fit and result in weird expected winning
% bids. With cleaned sample this should not result in more than 1 or 2 dropped lines.
cf_revenue_gross((cf_revenue_gross(:,2)>1000),:) = [];
% Drop weird lines that result in NaN expected bids. With cleaned sample, this should not result in any dropped lines.
check_nan_revenue = max(isnan(cf_revenue_cfnetgross(:,2)), isnan(cf_revenue_net(:,2)));
cf_revenue_net(check_nan_revenue,:) = [];
cf_revenue_cfnetgross(check_nan_revenue,:) = [];
cf_revenue_change_netgross = cf_revenue_cfnetgross - cf_revenue_net;
cf_revenue_change_ng_rel= (cf_revenue_cfnetgross - cf_revenue_net) ./ cf_revenue_net;
subplot(2,1,1)
hist(cf_revenue_change_ng_rel(:,2))
title('Histogram of changes in expected winning bid')
subplot(2,1,2)
hist(-cf_revenue_change_ng_rel(:,4))
title('Histogram of changes in agency payoff')
filename = project_paths('OUT_FIGURES','hist_rev_change');
saveas(gcf,filename,'pdf');

% Compute means of expected bids and agency payoff gain.
sprintf('Average winning bid, expected winning bid, expected revenue and agency payoff for gross auctions: \n')
cf_rev_gross_mean = mean(cf_revenue_gross)
cf_rev_gross_median = median(cf_revenue_gross)
sprintf('Average winning bid, expected winning bid, expected revenue and agency payoff for net auctions: \n')
cf_rev_net_mean = mean(cf_revenue_net)
cf_rev_net_median = median(cf_revenue_net)

sprintf('Average winning bid, expected winning bid, expected revenue and agency payoff for net auctions procured as gross auctions: \n')
cf_rev_cfnetgross_mean = mean(cf_revenue_cfnetgross)
cf_rev_cfnetgross_median = median(cf_revenue_cfnetgross)
% Manual fix for ensuring that after cleaning of outliers revenue
% statistics are exactly identical.
cf_rev_cfnetgross_mean(:,3) = cf_rev_net_mean(:,3);
cf_rev_cfnetgross_median(:,3) = cf_rev_net_median(:,3);
cf_rev_cfnetgross_mean
cf_rev_cfnetgross_median

% Combine revenue counterfactuals in one table.
results_cf_rev = [cf_rev_gross_mean; cf_rev_net_mean; cf_rev_cfnetgross_mean];
results_cf_rev_small = [cf_rev_gross_mean(:,1:2); cf_rev_net_mean(:,1:2); cf_rev_cfnetgross_mean(:,1:2)];
results_cf_rev_median = [cf_rev_gross_median; cf_rev_net_median; cf_rev_cfnetgross_median];
results_cf_rev_small_median = [cf_rev_gross_median(:,1:2); cf_rev_net_median(:,1:2); cf_rev_cfnetgross_median(:,1:2)];



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute average efficiency for different number of bidders.

%% Compute average statistics and format results.
% Record minimum number in first column, and maximum number in second
% column.
sum_Pr_win_gross_check = zeros(length(N_gross),2);
for t=1:length(N_gross)
    N_max_t = N_pot_gross(t); 
    sum_pr_gross_aux = sum_Pr_win_gross(t,1:N_max_t);
    sum_Pr_win_gross_check(t,1) = min(sum_pr_gross_aux);
    sum_Pr_win_gross_check(t,2) = max(sum_pr_gross_aux);
end
sum_Pr_win_net_check = zeros(length(N_net),2);
for t=1:length(N_net)
    N_max_t = N_pot_net(t); 
    sum_pr_net_aux = sum_Pr_win_net(t,1:N_max_t);
    sum_Pr_win_net_check(t,1) = min(sum_pr_net_aux);
    sum_Pr_win_net_check(t,2) = max(sum_pr_net_aux);
end
sum_Pr_win_cfnetgross_check = zeros(length(N_net),2);
for t=1:length(N_net)
    N_max_t = N_pot_net(t); 
    sum_pr_cfnetgross_aux = sum_Pr_win_cfnetgross(t,1:N_max_t);
    sum_Pr_win_cfnetgross_check(t,1) = min(sum_pr_cfnetgross_aux);
    sum_Pr_win_cfnetgross_check(t,2) = max(sum_pr_cfnetgross_aux);
end

% Make sure that lines for which counterfactuals failed to converge, are not taken into account.
% With cleaned sample, this should not drop any lines.
Pr_efficient_gross(sum_Pr_win_gross_check(:,1)<0.85 | sum_Pr_win_gross_check(:,2)>1.15)=[];
N_eff_gross = N_gross;
N_eff_gross(sum_Pr_win_gross_check<0.85 | sum_Pr_win_gross_check>1.15) = [];

% For gross auctions.
Pr_eff_gross_N = zeros(5,1);
for nb = 2:5
    % Select only auctions with nb bidders.
    Pr_eff_gross_aux = Pr_efficient_gross(N_eff_gross==nb);
    % Compute mean for this auction.
    Pr_eff_gross_N(nb-1,1) = mean(Pr_eff_gross_aux);
end
% Same thing for N>2.
% Select only auctions with more than 2 bidders.
Pr_eff_gross_aux = Pr_efficient_gross(N_eff_gross>2);
% Compute mean for this auction.
Pr_eff_gross_N(end,1) = mean(Pr_eff_gross_aux);


% For net auctions.
N_eff_net = N_net;
% Make sure that lines for which counterfactuals failed to converge, are not taken into account.
% With cleaned sample, this should not drop any lines.
N_eff_net(Pr_efficient_net<0.02) = [];
Pr_efficient_net(Pr_efficient_net<0.02) = [];

% Efficiency for net auctions by number of bidders.
Pr_eff_net_N = zeros(5,1);
for nb = 2:5
    % Select only auctiosn with nb bidders.
    Pr_eff_net_aux = Pr_efficient_net(N_eff_net==nb);
    % Compute mean for this auction.
    Pr_eff_net_N(nb-1,1) = mean(Pr_eff_net_aux);
end
% Same thing for N>2.
Pr_eff_net_aux = Pr_efficient_net(N_eff_net>2);
% Compute mean for this auction.
Pr_eff_net_N(end,1) = mean(Pr_eff_net_aux);

% For net auctions procured as gross.
N_eff_netgross = N_net;

% Efficiency for net auctions procured as gross by number of bidders.
Pr_eff_netgross_N = zeros(5,1);
for nb = 2:5
    % Select only auctiosn with nb bidders.
    Pr_eff_netgross_aux = Pr_efficient_cfnetgross(N_eff_netgross==nb);
    % Compute mean for this auction.
    Pr_eff_netgross_N(nb-1,1) = mean(Pr_eff_netgross_aux);
end
% Same thing for N>2.
% Select only auctions with more than 2 bidders.
Pr_eff_netgross_aux = Pr_efficient_cfnetgross(N_eff_netgross>2);
% Compute mean for this auction.
Pr_eff_netgross_N(end,1) = mean(Pr_eff_netgross_aux);

% Summarize evolution of different average efficiency levels as N
% increases.
Pr_eff_N = [Pr_eff_gross_N, Pr_eff_net_N, Pr_eff_netgross_N];
% Export data to csv file for reformatting in Python.
VarNames_eff_N = {'gross','net','cfnetgross'};
cf_eff_n_data = table(Pr_eff_N(:,1), Pr_eff_N(:,2), Pr_eff_N(:,3), ...
			  'VariableNames', VarNames_eff_N);
% Export efficiency data to to csv-file.
writetable(cf_eff_n_data, project_paths('OUT_ANALYSIS','cf_eff_data_n.csv'));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Conduct t-test of equality of mean efficiency.
% For gross vs. net (observed data).
[eff_data_dec, eff_data_p] = ttest2(Pr_efficient_gross,Pr_efficient_net);
[eff_cf_dec, eff_cf_p] = ttest2(Pr_efficient_gross,Pr_efficient_cfnetgross);
% For testing equality of median.
[eff_data_p_wilcoxon] = ranksum(Pr_efficient_gross,Pr_efficient_net);
[eff_cf_p_wilcoxon] = ranksum(Pr_efficient_gross,Pr_efficient_cfnetgross);
fprintf('P-value for testing equality of mean efficiency \n in observed data: %f \n', eff_data_p)
fprintf('P-value for testing equality of mean efficiency \n in counterfactual: %f \n', eff_cf_p)
fprintf('P-value for testing equality of median efficiency \n in observed data: %f \n', eff_data_p_wilcoxon)
fprintf('P-value for testing equality of median efficiency \n in counterfactual: %f \n', eff_cf_p_wilcoxon)

% Efficiency statistics for gross auction sample.
Pr_eff_gross_mean = mean(Pr_efficient_gross);
Pr_eff_gross_median = median(Pr_efficient_gross);

subplot(3,1,1)
hist(Pr_efficient_gross,10)
title('Histogram of efficiency probabilities in gross auctions','Interpreter', 'Latex', 'FontSize', 12)
xlim([0,1])
ylabel('Frequency','Interpreter','latex')
xlabel('Pr(selecting efficient firm','Interpreter','latex')

% Efficiency statistics for net auction sample.
Pr_eff_net_mean = mean(Pr_efficient_net);
Pr_eff_net_median = median(Pr_efficient_net);

subplot(3,1,2)
hist(Pr_efficient_net,8)
title('Histogram of efficiency probabilities in net auctions','Interpreter', 'Latex', 'FontSize', 12)
ylabel('Frequency','Interpreter','latex')
xlabel('Pr(selecting efficient firm','Interpreter','latex')
axis([0,1,0,12])

% For net auction sample procured as gross.
Pr_eff_cfnetgross_mean = mean(Pr_efficient_cfnetgross);
Pr_eff_cfnetgross_median = median(Pr_efficient_cfnetgross);

subplot(3,1,3)
hist(Pr_efficient_cfnetgross,17)
title('Histogram of counterfactual efficiency probabilities (net as gross)', 'Interpreter', 'Latex', 'FontSize', 12)
ylabel('Frequency','Interpreter','latex')
xlabel('Pr(selecting efficient firm','Interpreter','latex')
xlim([0,1])
filename = project_paths('OUT_FIGURES','hist_eff');
saveas(gcf,filename,'pdf');

% Same histogram for agency payoff.
sprintf('Agency revenues of gross auctions: \n')
subplot(3,1,1)
hist(cf_revenue_net(:,2),15)
title('Histogram of winning bids in net auctions','Interpreter', 'Latex', 'FontSize', 12)
xlim([0,40])
ylabel('Frequency','Interpreter','latex')
xlabel('Winning bid (in 10 Mio.~EUR)','Interpreter','latex')

sprintf('Agency revenues of net auctions: \n')
subplot(3,1,2)
hist(cf_revenue_cfnetgross(:,2),15)
title('Histogram of winning bids in net auctions procured as gross','Interpreter', 'Latex', 'FontSize', 12)
ylabel('Frequency','Interpreter','latex')
xlabel('Winning bid (in 10 Mio.~EUR)','Interpreter','latex')
xlim([0,40])
sprintf('Agency revenues of net auctions procured as gross: \n')
subplot(3,1,3)
hist((cf_revenue_cfnetgross(:,2)-cf_revenue_net(:,2))./cf_revenue_net(:,2),25)
title('Histogram of relative difference in winning bids when net procured as gross', 'Interpreter', 'Latex', 'FontSize', 12)
xlim([-1,2])
ylabel('Frequency','Interpreter','latex')
xlabel('Relative change of winning bid','Interpreter','latex')
filename = project_paths('OUT_FIGURES','hist_revenues');
saveas(gcf,filename,'pdf');

% Conduct t-test of equality of mean winning bid.
% For gross vs. net (observed data).
[wb_data_dec, wb_data_p] = ttest2(cf_revenue_gross(:,2),cf_revenue_net(:,2));
% For gross vs. counterfactual.
[wb_cf_dec, wb_cf_p] = ttest2(cf_revenue_gross(:,2),cf_revenue_cfnetgross(:,2));

% Write new log-file.
fid = fopen(project_paths('OUT_ANALYSIS','counterfactuals_eff_endo_N.log'),'wt');
fprintf(fid,strcat('Summary of counterfactual efficiency probabilities WITH ENDOGENOUS N\nEstimated on:\n', date,'\n\n'));

fprintf(fid,'Efficiency probability of gross auctions: \n Mean efficiency probability: %6.4f \n Median efficiency probability: %6.4f \n Number of lines for which computation worked well: %d. \n\n', ...
        Pr_eff_gross_mean,Pr_eff_gross_median, length(Pr_efficient_gross));

fprintf(fid,'Efficiency probability of net auctions: \n Mean efficiency probability: %6.4f \n Median efficiency probability: %6.4f \n Number of lines for which computation worked well: %d. \n\n', ...
        Pr_eff_net_mean,Pr_eff_net_median, length(Pr_efficient_net));
   
fprintf(fid,'Efficiency probability of net auctions procured as gross: \n Mean efficiency probability: %6.4f \n Median efficiency probability: %6.4f \n Number of lines for which computation worked well: %d. \n\n', ...
        Pr_eff_cfnetgross_mean,Pr_eff_cfnetgross_median, length(Pr_efficient_cfnetgross));
   
fprintf(fid,'\nP-value for testing equality of mean efficiency in observed data: %6.4f \n', eff_data_p);
fprintf(fid,'P-value for testing equality of mean efficiency in counterfactual: %6.4f \n', eff_cf_p);
fprintf(fid,'P-value for testing equality of median efficiency in observed data: %6.4f \n', eff_data_p_wilcoxon);
fprintf(fid,'P-value for testing equality of median efficiency in counterfactual: %6.4f \n\n', eff_cf_p_wilcoxon);
fprintf(fid,'P-value for testing equality of mean winning bis in observed data: %6.4f \n', wb_data_p);
fprintf(fid,'P-value for testing equality of mean winning bids in counterfactual: %6.4f \n', wb_cf_p);
fclose(fid);

%% Prepare data on efficiency probabilities for formatting in Python.
results_cf_eff = [Pr_eff_gross_mean, Pr_eff_net_mean, Pr_eff_cfnetgross_mean; ...
                  Pr_eff_gross_median, Pr_eff_net_median, Pr_eff_cfnetgross_median];

col_labels = {'Gross Auctions', 'Net Auctions', 'Net $\rightarrow$ Gross'};
row_labels = {'Pr(selecting efficient firm) - mean';'Pr(selecting efficient firm) - median'};
% Export table directly (not used in final paper).
matrix2latex(results_cf_eff,project_paths('OUT_TABLES','resultscfeff.tex'),'headerRow', col_labels, 'headerColumn', row_labels, 'format', '%6.4f', 'caption', 'Efficiency comparison for different auction formats (endogenous $N$)', 'label', 'cfresultseff');


% Export data to csv file for reformatting in Python.
VarNames_eff = {'gross','net','cfnetgross'};
cf_eff_data = table(results_cf_eff(:,1), results_cf_eff(:,2), results_cf_eff(:,3), ...
			  'VariableNames', VarNames_eff);
% This file contains summary efficiency statistics for the case with
% endogenous N.
writetable(cf_eff_data, project_paths('OUT_ANALYSIS','cf_eff_data.csv'));


%% Format results in table (currently only with efficiency probabilities, potentially supplemented with revenuecomparison etc infuture)
col_labels = {'Observed bid', 'Predicted bid', 'E(ticket revenue)', 'E(agency payoff)'};
row_labels = {'Gross auctions','Net auctions', 'Net $\rightarrow$ Gross auctions'};
matrix2latex(results_cf_rev,project_paths('OUT_TABLES','resultscfrev.tex'),'headerRow', col_labels, 'headerColumn', row_labels, 'format', '%6.4f', 'caption', 'Revenue comparison for different auction formats', 'label', 'cfresultsrev');
% Export data to csv file for reformatting in Python.
VarNames_rev = {'bid_obs','bid_pred','mean_rev','e_payoff'};
cf_rev_data = table(results_cf_rev(:,1),results_cf_rev(:,2), results_cf_rev(:,3), results_cf_rev(:,4),  ...
			  'VariableNames', VarNames_rev);
writetable(cf_rev_data, project_paths('OUT_ANALYSIS','cf_rev_data.csv'));
% Write same variables, but median across auctions, to file.
VarNames_rev = {'bid_obs','bid_pred','median_rev','median_payoff'};
cf_rev_data_median = table(results_cf_rev_median(:,1),results_cf_rev_median(:,2), results_cf_rev_median(:,3), results_cf_rev_median(:,4),  ...
			  'VariableNames', VarNames_rev);
writetable(cf_rev_data_median, project_paths('OUT_ANALYSIS','cf_rev_data_median.csv'));



% Small version of revenue table.
%% Format results in table (currently only with efficiency probabilities, potentially supplemented with revenuecomparison etc infuture)
col_labels = {'Observed bid', 'Predicted bid'};
row_labels = {'Gross auctions','Net auctions', 'Net $\rightarrow$ Gross auctions'};
matrix2latex(results_cf_rev_small,project_paths('OUT_TABLES','resultscfrevsmall.tex'),'headerRow', col_labels, 'headerColumn', row_labels, 'format', '%6.4f', 'caption', 'Winning bid comparison for different auction formats', 'label', 'cfresultsrev');


% Plot distribution of relative efficiency changes.
subplot(1,1,1)
% Safety check for unreasonably large efficiency gains due to numerical
% imprecision on some lines. Shouldn't be an issue with cleaned sample.
rel_eff_change_netgross(rel_eff_change_netgross>10) = [];

% Compute average efficiency gain.
mean_rel_eff_gain = mean(rel_eff_change_netgross(rel_eff_change_netgross>0));
mean_rel_eff_loss = mean(rel_eff_change_netgross(rel_eff_change_netgross<0));

hist(rel_eff_change_netgross)
title('Histogram of relative change in efficiency probability (net -> gross)')
filename = project_paths('OUT_FIGURES','hist_rel_eff_gain');
saveas(gcf,filename,'pdf');
% END OF FORMATTING RESULTS FOR ENDOGENOUS N
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save post-counterfactual workspace.
save(project_paths('OUT_ANALYSIS','postcounterfactual_workspace'));




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% START OF FORMATTING RESULTS FOR CONSTANT N
% Do similar safety checks for non-converging counterfactuals as we did above for endogenous N.
disp('---ANALYZE RESULTS FOR CONSTANT NUMBER OF BIDDERS---');
% Compute relative efficiency changes when going from gross to net.
rel_eff_change_netgross = (Pr_efficient_cfnetgross_constant_n - Pr_efficient_net_constant_n) ./ Pr_efficient_net_constant_n;

% Count efficiency gains and losses.
eff_change_netgross  = rel_eff_change_netgross .* Pr_efficient_net_constant_n;
sum_eff_gain = sum(rel_eff_change_netgross>=0) ./ length(Pr_efficient_cfnetgross_constant_n)
sum_eff_loss = sum(rel_eff_change_netgross<0) ./ length(Pr_efficient_cfnetgross_constant_n)
% Compute mean efficiency gain.
mean_eff_loss = mean(eff_change_netgross(eff_change_netgross<0))
mean_eff_gain = mean(eff_change_netgross(eff_change_netgross>0 & eff_change_netgross<=1))
sprintf('Share of net tracks for which gross leads to efficiency gain (constant N): %.2f \n', sum_eff_gain)
sprintf('Share of net tracks for which gross leads to efficiency loss (constant N): %.2f \n', sum_eff_loss)
sprintf('Average efficiency gain when net tracks procured as gross (constant N): %.2f \n', mean_eff_gain)
sprintf('Average efficiency loss when net tracks procured as gross (constant N): %.2f \n', mean_eff_loss)
%% Compare revenues and agency payoffs first.
cf_revenue_gross_constant_n((cf_revenue_gross_constant_n(:,2)>1000),:) = [];
check_nan_revenue = max(isnan(cf_revenue_cfnetgross_constant_n(:,2)), isnan(cf_revenue_net_constant_n(:,2)));
cf_revenue_net_constant_n(check_nan_revenue,:) = [];
cf_revenue_cfnetgross_constant_n(check_nan_revenue,:) = [];
cf_revenue_change_netgross = cf_revenue_cfnetgross_constant_n - cf_revenue_net_constant_n;
cf_revenue_change_ng_rel= (cf_revenue_cfnetgross_constant_n - cf_revenue_net_constant_n) ./ cf_revenue_net_constant_n;
subplot(2,1,1)
hist(cf_revenue_change_ng_rel(:,2))
title('Histogram of changes in expected winning bid (constant N)')
subplot(2,1,2)
hist(-cf_revenue_change_ng_rel(:,4))
title('Histogram of changes in agency payoff (constant N)')
filename = project_paths('OUT_FIGURES','hist_rev_change_cons_n');
saveas(gcf,filename,'pdf');
% Compute means of expected bids and agency payoff gain.
sprintf('Average winning bid, expected winning bid, expected revenue and agency payoff for gross auctions (constant N): \n')
cf_rev_gross_mean = mean(cf_revenue_gross_constant_n)
cf_rev_gross_median = median(cf_revenue_gross_constant_n)

sprintf('Average winning bid, expected winning bid, expected revenue and agency payoff for net auctions (constant N): \n')
cf_rev_net_mean = mean(cf_revenue_net_constant_n)
cf_rev_net_median = median(cf_revenue_net_constant_n)

sprintf('Average winning bid, expected winning bid, expected revenue and agency payoff for net auctions procured as gross auctions (constant N): \n')
cf_rev_cfnetgross_mean = mean(cf_revenue_cfnetgross_constant_n)
% Compute same statistics for median.
cf_rev_cfnetgross_median = median(cf_revenue_cfnetgross_constant_n)
cf_rev_cfnetgross_mean(:,3) = cf_rev_net_mean(:,3);
cf_rev_cfnetgross_median(:,3) = cf_rev_net_median(:,3);
cf_rev_cfnetgross_mean
cf_rev_cfnetgross_median

% Combine revenue counterfactuals in one table.
results_cf_rev = [cf_rev_gross_mean; cf_rev_net_mean; cf_rev_cfnetgross_mean];
results_cf_rev_small = [cf_rev_gross_mean(:,1:2); cf_rev_net_mean(:,1:2); cf_rev_cfnetgross_mean(:,1:2)];
results_cf_rev_median = [cf_rev_gross_median; cf_rev_net_median; cf_rev_cfnetgross_median];
results_cf_rev_small_median = [cf_rev_gross_median(:,1:2); cf_rev_net_median(:,1:2); cf_rev_cfnetgross_median(:,1:2)];



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute average efficiency for different number of bidders.
%% Compute some average statistics and format results.
Pr_efficient_gross_constant_n(sum_Pr_win_gross_constant_n<0.85 | sum_Pr_win_gross_constant_n>1.15)=[];
N_eff_gross = N_gross;
N_eff_gross(sum_Pr_win_gross_constant_n<0.85 | sum_Pr_win_gross_constant_n>1.15) = [];

% For gross auctions.
Pr_eff_gross_N = zeros(5,1);
for nb = 2:5
    % Select only auctiosn with nb bidders.
    Pr_eff_gross_aux = Pr_efficient_gross_constant_n(N_eff_gross==nb);
    % Compute mean for this auction.
    Pr_eff_gross_N(nb-1,1) = mean(Pr_eff_gross_aux);
end
% Same thing for N>2.
% Select only auctions with more than 2 bidders.
Pr_eff_gross_aux = Pr_efficient_gross_constant_n(N_eff_gross>2);
% Compute mean for this auction.
Pr_eff_gross_N(end,1) = mean(Pr_eff_gross_aux);


% For net auctions.
N_eff_net = N_net;
% Efficiency for net auctions by number of bidders.
Pr_eff_net_N = zeros(5,1);
for nb = 2:5
    % Select only auctiosn with nb bidders.
    Pr_eff_net_aux = Pr_efficient_net_constant_n(N_eff_net==nb);
    % Compute mean for this auction.
    Pr_eff_net_N(nb-1,1) = mean(Pr_eff_net_aux);
end
% Same thing for N>2.
% Select only auctions with more than 2 bidders.
Pr_eff_net_aux = Pr_efficient_net_constant_n(N_eff_net>2);
% Compute mean for this auction.
Pr_eff_net_N(end,1) = mean(Pr_eff_net_aux);

% For net auctions procured as gross.
Pr_efficient_cfnetgross_constant_n(sum_Pr_win_cfnetgross_constant_n<0.85 | sum_Pr_win_cfnetgross_constant_n>1.15)=[];
N_eff_netgross = N_net;
N_eff_netgross(sum_Pr_win_cfnetgross_constant_n<0.85 | sum_Pr_win_cfnetgross_constant_n>1.15) = [];

% Efficiency for net auctions procured as gross by number of bidders.
Pr_eff_netgross_N = zeros(5,1);
for nb = 2:5
    % Select only auctiosn with nb bidders.
    Pr_eff_netgross_aux = Pr_efficient_cfnetgross_constant_n(N_eff_netgross==nb);
    % Compute mean for this auction.
    Pr_eff_netgross_N(nb-1,1) = mean(Pr_eff_netgross_aux);
end
% Same thing for N>2.
% Select only auctions with more than 2 bidders.
Pr_eff_netgross_aux = Pr_efficient_cfnetgross_constant_n(N_eff_netgross>2);
% Compute mean for this auction.
Pr_eff_netgross_N(end,1) = mean(Pr_eff_netgross_aux);

% Summarize evolution of different average efficiency levels as N
% increases.
Pr_eff_N = [Pr_eff_gross_N, Pr_eff_net_N, Pr_eff_netgross_N];
% Export data to csv file for reformatting in Python.
VarNames_eff_N = {'gross','net','cfnetgross'};
cf_eff_n_data = table(Pr_eff_N(:,1), Pr_eff_N(:,2), Pr_eff_N(:,3), ...
			  'VariableNames', VarNames_eff_N);
% Export efficiency data to to csv-file.
writetable(cf_eff_n_data, project_paths('OUT_ANALYSIS','cf_eff_data_n_cons_n.csv'));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Conduct t-test of equality of mean efficiency.
% For gross vs. net (observed data).
[eff_data_dec, eff_data_p_cons_n] = ttest2(Pr_efficient_gross_constant_n,Pr_efficient_net_constant_n);
[eff_cf_dec, eff_cf_p_cons_n] = ttest2(Pr_efficient_gross_constant_n,Pr_efficient_cfnetgross_constant_n);
[eff_data_p_wilcoxon_cons_n] = ranksum(Pr_efficient_gross_constant_n,Pr_efficient_net_constant_n);
[eff_cf_p_wilcoxon_cons_n] = ranksum(Pr_efficient_gross_constant_n,Pr_efficient_cfnetgross_constant_n);
fprintf('P-value for testing equality of mean efficiency \n in observed data (constant N): %f \n', eff_data_p_cons_n);
fprintf('P-value for testing equality of mean efficiency \n in counterfactual (constant N): %f \n', eff_cf_p_cons_n);
fprintf('P-value for testing equality of median efficiency \n in observed data (constant N): %f \n', eff_data_p_wilcoxon_cons_n);
fprintf('P-value for testing equality of median efficiency \n in counterfactual (constant N): %f \n', eff_cf_p_wilcoxon_cons_n);

% Efficiency statistics for gross auction sample.
Pr_eff_gross_cons_n_mean = mean(Pr_efficient_gross_constant_n);
Pr_eff_gross_cons_n_median = median(Pr_efficient_gross_constant_n);

subplot(3,1,1)
hist(Pr_efficient_gross_constant_n,10)
title('Histogram of efficiency probabilities in gross auctions (constant N)','Interpreter', 'Latex', 'FontSize', 12)
xlim([0,1])
ylabel('Frequency','Interpreter','latex')
xlabel('Pr(selecting efficient firm','Interpreter','latex')
% Efficiency statistics for net auction sample.
Pr_eff_net_cons_n_mean = mean(Pr_efficient_net_constant_n);
Pr_eff_net_cons_n_median = median(Pr_efficient_net_constant_n);

subplot(3,1,2)
hist(Pr_efficient_net_constant_n,8)
title('Histogram of efficiency probabilities in net auctions (constant N)','Interpreter', 'Latex', 'FontSize', 12)
ylabel('Frequency','Interpreter','latex')
xlabel('Pr(selecting efficient firm','Interpreter','latex')
axis([0,1,0,12])
% For net auction sample procured as gross.
Pr_eff_cfnetgross_cons_n_mean = mean(Pr_efficient_cfnetgross_constant_n);
Pr_eff_cfnetgross_cons_n_median = median(Pr_efficient_cfnetgross_constant_n);

subplot(3,1,3)
hist(Pr_efficient_cfnetgross_constant_n,17)
title('Histogram of counterfactual efficiency probabilities (net as gross, constant N)', 'Interpreter', 'Latex', 'FontSize', 12)
ylabel('Frequency','Interpreter','latex')
xlabel('Pr(selecting efficient firm','Interpreter','latex')
xlim([0,1])
filename = project_paths('OUT_FIGURES','hist_eff_cons_n');
saveas(gcf,filename,'pdf');

% Same histogram for agency payoff.
sprintf('Agency revenues of gross auctions (constant N): \n')
subplot(3,1,1)
hist(cf_revenue_net_constant_n(:,2),15)
title('Histogram of winning bids in net auctions (constant N)','Interpreter', 'Latex', 'FontSize', 12)
xlim([0,40])
ylabel('Frequency','Interpreter','latex')
xlabel('Winning bid (in 10 Mio.~EUR)','Interpreter','latex')

sprintf('Agency revenues of net auctions (constant N): \n')
subplot(3,1,2)
hist(cf_revenue_cfnetgross_constant_n(:,2),15)
title('Histogram of winning bids in net auctions procured as gross (constant N)','Interpreter', 'Latex', 'FontSize', 12)
ylabel('Frequency','Interpreter','latex')
xlabel('Winning bid (in 10 Mio.~EUR)','Interpreter','latex')
xlim([0,40])
sprintf('Agency revenues of net auctions procured as gross (constant N): \n')
subplot(3,1,3)
hist((cf_revenue_cfnetgross_constant_n(:,2)-cf_revenue_net_constant_n(:,2))./cf_revenue_net_constant_n(:,2),25)
title('Histogram of relative difference in winning bids when net procured as gross (constant N)', 'Interpreter', 'Latex', 'FontSize', 12)
xlim([-1,2])
ylabel('Frequency','Interpreter','latex')
xlabel('Relative change of winning bid','Interpreter','latex')
filename = project_paths('OUT_FIGURES','hist_revenues_cons_n');
saveas(gcf,filename,'pdf');

% Conduct t-test of equality of mean winning bid.
% For gross vs. net (observed data).
[wb_data_dec, wb_data_p_cons_n] = ttest2(cf_revenue_gross_constant_n(:,2),cf_revenue_net_constant_n(:,2));
% For gross vs. counterfactual.
[wb_cf_dec, wb_cf_p_cons_n] = ttest2(cf_revenue_gross_constant_n(:,2),cf_revenue_cfnetgross_constant_n(:,2));

% Write new log-file.
%% Write some entry cost estimates to txt file.
fid = fopen(project_paths('OUT_ANALYSIS','counterfactuals_constant_n.log'),'wt');
fprintf(fid,strcat('Summary of counterfactual efficiency probabilities (constant N)\nEstimated on:\n', date,'\n\n'));

fprintf(fid,'Efficiency probability of gross auctions (constant N): \n Mean efficiency probability: %6.4f \n Median efficiency probability: %6.4f \n Number of lines for which computation worked well: %d. \n\n', ...
        Pr_eff_gross_mean,Pr_eff_gross_median, length(Pr_efficient_gross_constant_n));

fprintf(fid,'Efficiency probability of net auctions (constant N): \n Mean efficiency probability: %6.4f \n Median efficiency probability: %6.4f \n Number of lines for which computation worked well: %d. \n\n', ...
        Pr_eff_net_mean,Pr_eff_net_median, length(Pr_efficient_net_constant_n));
   
fprintf(fid,'Efficiency probability of net auctions procured as gross (constant N): \n Mean efficiency probability: %6.4f \n Median efficiency probability: %6.4f \n Number of lines for which computation worked well: %d. \n\n', ...
        Pr_eff_cfnetgross_mean,Pr_eff_cfnetgross_median, length(Pr_efficient_cfnetgross_constant_n));
   
fprintf(fid,'\nP-value for testing equality of mean efficiency in observed data (constant N): %6.4f \n', eff_data_p);
fprintf(fid,'P-value for testing equality of mean efficiency in counterfactual (constant N): %6.4f \n', eff_cf_p);
fprintf(fid,'P-value for testing equality of median efficiency in observed data (constant N): %6.4f \n', eff_data_p_wilcoxon);
fprintf(fid,'P-value for testing equality of median efficiency in counterfactual (constant N): %6.4f \n\n', eff_cf_p_wilcoxon);
fprintf(fid,'P-value for testing equality of mean winning bis in observed data (constant N): %6.4f \n', wb_data_p);
fprintf(fid,'P-value for testing equality of mean winning bids in counterfactual (constant N): %6.4f \n', wb_cf_p);
fclose(fid);

%% Prepare data on efficiency probabilities for formatting in Python.
results_cf_eff = [Pr_eff_gross_cons_n_mean, Pr_eff_net_cons_n_mean, Pr_eff_cfnetgross_cons_n_mean; ...
                  Pr_eff_gross_cons_n_median, Pr_eff_net_cons_n_median, Pr_eff_cfnetgross_cons_n_median];

col_labels = {'Gross Auctions', 'Net Auctions', 'Net $\rightarrow$ Gross'};
row_labels = {'Pr(selecting efficient firm) - mean';'Pr(selecting efficient firm) - median'};
% Export table directly (not used in final paper).
matrix2latex(results_cf_eff,project_paths('OUT_TABLES','resultscfeffconsn.tex'),'headerRow', col_labels, 'headerColumn', row_labels, 'format', '%6.4f', 'caption', 'Efficiency comparison for different auction formats (constant $N$)', 'label', 'cfresultseffconsn');

% Export data to csv file for reformatting in Python.
VarNames_eff = {'gross','net','cfnetgross'};
cf_eff_data = table(results_cf_eff(:,1), results_cf_eff(:,2), results_cf_eff(:,3), ...
			  'VariableNames', VarNames_eff);
writetable(cf_eff_data, project_paths('OUT_ANALYSIS','cf_eff_data_cons_n.csv'));


%% Format results in table (currently only with efficiency probabilities, potentially supplemented with revenuecomparison etc infuture)
col_labels = {'Observed bid', 'Predicted bid', 'E(ticket revenue)', 'E(agency payoff)'};
row_labels = {'Gross auctions','Net auctions', 'Net $\rightarrow$ Gross auctions'};

matrix2latex(results_cf_rev,project_paths('OUT_TABLES','resultscfrevconsn.tex'),'headerRow', col_labels, 'headerColumn', row_labels, 'format', '%6.4f', 'caption', 'Revenue comparison for different auction formats (constant N)', 'label', 'cfresultsrev');
% Export data to csv file for reformatting in Python.
VarNames_rev = {'bid_obs','bid_pred','mean_rev','e_payoff'};
cf_rev_data = table(results_cf_rev(:,1),results_cf_rev(:,2), results_cf_rev(:,3), results_cf_rev(:,4),  ...
			  'VariableNames', VarNames_rev);
% Export counterfactual revenue data to csv-file for formatting in Python.
writetable(cf_rev_data, project_paths('OUT_ANALYSIS','cf_rev_data_cons_n.csv'));
% Write same variables, but median across auctions, to file.
VarNames_rev = {'bid_obs','bid_pred','median_rev','median_payoff'};
cf_rev_data_median = table(results_cf_rev_median(:,1),results_cf_rev_median(:,2), results_cf_rev_median(:,3), results_cf_rev_median(:,4),  ...
			  'VariableNames', VarNames_rev);
writetable(cf_rev_data_median, project_paths('OUT_ANALYSIS','cf_rev_data_median_cons_n.csv'));



% Small version of revenue table.
%% Format results in table (currently only with efficiency probabilities, potentially supplemented with revenue comparison etc in future)
col_labels = {'Observed bid', 'Predicted bid'};
row_labels = {'Gross auctions','Net auctions', 'Net $\rightarrow$ Gross auctions'};
matrix2latex(results_cf_rev_small,project_paths('OUT_TABLES','resultscfrevsmall.tex'),'headerRow', col_labels, 'headerColumn', row_labels, 'format', '%6.4f', 'caption', 'Winning bid comparison for different auction formats', 'label', 'cfresultsrev');

% Plot distribution of relative efficiency changes.
subplot(1,1,1)
rel_eff_change_netgross(rel_eff_change_netgross>10) = [];
% Compute average efficiency gain.
mean_rel_eff_gain = mean(rel_eff_change_netgross(rel_eff_change_netgross>0));
mean_rel_eff_loss = mean(rel_eff_change_netgross(rel_eff_change_netgross<0));
hist(rel_eff_change_netgross)
title('Histogram of relative change in efficiency probability (net -> gross)')
filename = project_paths('OUT_FIGURES','hist_rel_eff_gain_cons_n');
saveas(gcf,filename,'pdf');
% END OF FORMATTING RESULTS FOR CONSTANT N
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Save post-counterfactual workspace.
save(project_paths('OUT_ANALYSIS','postcounterfactual_workspace_constant_n'));
