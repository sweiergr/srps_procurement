%{
    Plot histogram of relative cost advantage jointly for net and gross
    sample.

%}

clear;
clc
%% Load data for gross and net samples.
load(project_paths('OUT_ANALYSIS','rca_data_gross.mat'))
load(project_paths('OUT_ANALYSIS','rca_data_net.mat'))

figuresize(15,12,'cm')
% Set cost threshold at which to truncate incumbent's cost advantage.
% For now: define it at 10 times the incumbent's median cost.
cost_threshold = 10;

% join two samples into one matrix.
median_costs = [median_costs_gross; median_costs_net];
rel_cost_adv_med = [rel_cost_adv_med_gross; rel_cost_adv_med_net];
mean_costs = [mean_costs_gross; mean_costs_net];
rel_cost_adv = [rel_cost_adv_gross; rel_cost_adv_net];
saveas(gcf,project_paths('OUT_FIGURES','rca_hist_combined_median'),'pdf');

set(gca,'TickLabelInterpreter','latex')
rel_cost_adv_med = (median_costs(:,2) - median_costs(:,1)) ./ abs(median_costs(:,1));
rel_cost_adv_med(rel_cost_adv_med>cost_threshold) = cost_threshold;
subplot(2,1,1)
hist(rel_cost_adv_med,75)
axis([-1,cost_threshold, 0, 15])
title('Relative cost advantage of incumbent (median) - histogram', 'Interpreter', 'Latex', 'FontSize', 12)
ylabel('Frequency', 'FontSize', 12, 'Interpreter', 'Latex');

% Median costs for each contract and bidder type.
% Incumbent in first column, entrant in second column.
set(gca,'TickLabelInterpreter','latex')
rel_cost_adv_med = (median_costs(:,2) - median_costs(:,1)) ./ abs(median_costs(:,1));
rel_cost_adv_med(rel_cost_adv_med>cost_threshold) = cost_threshold;
subplot(2,1,1)
hist(rel_cost_adv_med,75)
axis([-1,cost_threshold, 0, 15])
title('Relative cost advantage of incumbent (median) - histogram', 'Interpreter', 'Latex', 'FontSize', 12)
ylabel('Frequency', 'FontSize', 12, 'Interpreter', 'Latex');

rel_cost_adv = (mean_costs(:,2) - mean_costs(:,1)) ./ abs(mean_costs(:,1));
rel_cost_adv(rel_cost_adv>cost_threshold) = cost_threshold;
subplot(2,1,2)
hist(rel_cost_adv,75)
axis([-1,cost_threshold,0,15])
title('Relative cost advantage of incumbent (mean) - histogram', 'Interpreter', 'Latex', 'FontSize', 12)
ylabel('Frequency', 'FontSize', 12, 'Interpreter', 'Latex');

saveas(gcf,project_paths('OUT_FIGURES','rca_hist_combined'),'pdf');

% Same histogram
subplot(1,1,1)
hist(rel_cost_adv_med,50)
axis([-1,cost_threshold, 0, 22])
title('Relative (median) cost advantage of incumbent  - histogram', 'Interpreter', 'Latex', 'FontSize', 12)
ylabel('Frequency', 'FontSize', 12, 'Interpreter', 'Latex');
saveas(gcf,project_paths('OUT_FIGURES','rca_hist_combined_median'),'pdf');



