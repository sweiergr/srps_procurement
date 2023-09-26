%{
    Compute ratio of expected revenues over expected costs including entry
    costs.
    This is just for data exploration and not used in the IJIO-version of the paper.

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

% Load entry cost estimates for both sample.
load(project_paths('OUT_ANALYSIS','grossauction_workspace_entry'));
kappa_E_gross = kappa_E;
clearvars -except kappa_E_gross
load(project_paths('OUT_ANALYSIS','netauction_workspace_entry'));
kappa_E_net = kappa_E;
clearvars -except kappa_E_gross kappa_E_net

% Load postestimation workspace.
load(project_paths('OUT_ANALYSIS','postestimation_workspace_net'));

% Compute expected total costs including entry costs.
% Entry cost paid y both incumbent and entrants.
mean_costs_total = mean_costs + kappa_E_net;
% Entry cost only paid by entrants (this is probably what we should do).
mean_costs_total = mean_costs + [zeros(T,1), kappa_E_net];
% Just for debuggging: no entry costs at all.
% mean_costs_total = mean_costs;
% Compute share of mean revenues to mean costs.
share_rev_costtotal = repmat(mean_revenue,1,2) ./ mean_costs_total;
% Alternatively, compute mean of rev-cost-share,
% after outliers are kicked out.
share_rev_cost_clean = share_rev_costtotal;
share_rev_cost_clean(share_rev_cost_clean>10) = NaN;
% Compute median rev-cost-share, i.e., how much of costs can be covered by
% revenues.
median_share_rev_cost = nanmedian(share_rev_costtotal)
mean_share_rev_cost = nanmean(share_rev_costtotal)
median_share_rev_cost_clean = nanmedian(share_rev_cost_clean)
mean_share_rev_cost_clean = nanmean(share_rev_cost_clean)

% Combine over winner identities.
share_win = share_rev_cost_clean(:,1) .* db_win + share_rev_cost_clean(:,2) .* (1-db_win);
median_share_win_clean = nanmedian(share_win)
mean_share_win_cost_clean = nanmean(share_win)
