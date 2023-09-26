%{
    Simulate new net -> gross counterfactuals.
    1. Simulate net auction sample efficiency when no strategic bidding at
    all to isolate effect of noise through additional signal.
    2. Simulate net auction sample efficiency when bidders add gross
    markups to their combined net cost signal.
    
    Moreover, the file computes average markups for different bidder types 
    for when net auction sample is procured as gross. 

%}

clear
clc
format('short');
% Define necessary globals.
global data N_obs K
% Set seed in case any simulation is used.
rng(123456);
% Load gross auction estimation workspace.
load(project_paths('OUT_ANALYSIS','net_cost_estimation'));
% Load revenue data from net revenue estimation workspace.
load(project_paths('OUT_ANALYSIS','postestimation_workspace_net'));
% Construct containers for average markups for each auction and bidder
% type.
MU_average = zeros(N_obs,3);
% Set number of simulations for non-strategic revenue signal
% counterfactual.
NS_cf = 5000;
% Initialize container for counterfactual efficiency probabilities.
% When firms bid same markups as in gross.
eff_prob_cf_netgross_2 = zeros(T,1);
% When firms do not bid strategicaly at all.
eff_prob_cf_netgross_3 = zeros(T,1);


% Extract cost densities.
cost_dens_I = kdens_container(:,:,1,1);
cost_dens_E = kdens_container(:,:,2,1);
cost_cdf_I = cumsum(cost_dens_I,1) ./sum(cost_dens_I,1);
cost_cdf_E = cumsum(cost_dens_E,1) ./sum(cost_dens_E,1);

% Loop over all gross auctions.
for t=1:N_obs
    % Safety check to exclude weirdly behaving lines.
    if t>0 % Case by case debugging for the following lines: t~=39 & t~=60 & t~=66 & t~=67 & t~=69 & t~=73 & t~=74 & t~=75
    N_t = N(t);
    
    c_cdf_I = cost_cdf_I(:,t);
    c_cdf_E = cost_cdf_E(:,t);
    c_supp_I = cost_support_I(:,t);
    c_supp_E = cost_support_E(:,t);
    % Throughout set truncation bid to zero.
    truncation_bid = 0;
    
    % Extract bid function parameters for line t.
    lambda_I_sim = lambda_I_ng(t,1);
    lambda_E_sim = lambda_E_ng(t,1);
    rho_I_sim = rho_I_ng(t,1);
    rho_E_sim = rho_E_ng(t,1);

    % Construct bid grid that covers full range of bid distribution from 0.05
    % to 99.5 percent.
    % How many grid points to discretize to.
    n_grid = 1000;
    % Define lower and upper bounds of bid distribution to be integrated
    % over.
    bid_I_range = wblinv([0.005;0.995],lambda_I_sim,rho_I_sim);
    bid_E_range = wblinv([0.005;0.995],lambda_E_sim,rho_E_sim);
    bid_grid_I = linspace(bid_I_range(1),bid_I_range(2),n_grid);
    bid_grid_E = linspace(bid_E_range(1),bid_E_range(2),n_grid);

    % Compute markup terms for incumbent.
    % Compute CDF and PDF for both types at incumbent's simulated bid.
    [CDF_I, PDF_I] = eval_bf(lambda_I_sim,rho_I_sim,bid_grid_I,truncation_bid);
    [CDF_E, PDF_E] = eval_bf(lambda_E_sim,rho_E_sim,bid_grid_I,truncation_bid);
    % Numerator of incumbent's markup term.
    G_MI = (1-CDF_E).^(N(t)-1);
    % Denominator of entrant's markup term.
    g_MI = (N(t)-1) .* (1-CDF_E).^(N(t)-2) .* PDF_E;
    % Compute incumbent's markup.
    MU_I = G_MI ./ g_MI;
    % Set markups on right tail of bid to zero.
    MU_I(G_MI==0) = 0;
    % Compute vector of incumbent's simulated costs.
    cost_I_grid = bid_grid_I - MU_I;
    test_pdf_I = trapz(bid_grid_I,PDF_I);
    % Integrate over bid density.
    MU_average(t,1) = trapz(bid_grid_I,PDF_I.*MU_I);
    % Compute markup terms for entrants.
    % Compute CDF and PDF at entrants' simulated bid.
    [CDF_I, PDF_I] = eval_bf(lambda_I_sim,rho_I_sim,bid_grid_E,truncation_bid);
    [CDF_E, PDF_E] = eval_bf(lambda_E_sim,rho_E_sim,bid_grid_E, truncation_bid);
    % Numerator of entrant's markup term.
    G_ME = (1-CDF_E).^(N(t)-2) .* (1-CDF_I);
    % Denominator of entrnat's markup term.
    g_ME = (N(t)-2) .* (1-CDF_E).^(N(t)-3) .* PDF_E  .* (1-CDF_I) + PDF_I .* (1-CDF_E) .^ (N(t)-2);
    % Compute markup for entrant.
    MU_E = G_ME ./ g_ME;
    % Set markups on right tail of bid to zero.
    MU_E(G_ME==0) = 0;
    % Compute vector of entrants' costs.
    cost_E_grid = bid_grid_E - MU_E;
    test_pdf_E = trapz(bid_grid_E,PDF_E);
    % Integrate over bid density.
    MU_average(t,2) = trapz(bid_grid_E,PDF_E.*MU_E);
    
    % Draw from non-parametric cost distributions for entrant and incumbent.
    costs_sim_uni_I = rand(NS_cf,1);
    costs_sim_uni_E = rand(NS_cf,N_t-1);
    costs_sim = zeros(NS_cf,N_t);
    for nn=1:NS_cf
        uni_draw_I = costs_sim_uni_I(nn,1);
        uni_draw_E = costs_sim_uni_E(nn,:);
        % Determine cost position in cdf based on uniform draw.
        [diff_draw_I c_idx_I] = min(abs(uni_draw_I - c_cdf_I),[],1);
        [diff_draw_E c_idx_E] = min(abs(uni_draw_E - c_cdf_E),[],1);
        % Draw cost based on cdf index above.
        costs_sim(nn,1) = c_supp_I(c_idx_I);
        costs_sim(nn,2:end) = c_supp_E(c_idx_E);
    end
    
    % Draw revenue signal from trunacted normal distribution.
    % Generate parent normal distribution.
    pd_rev_parent = makedist('Normal',mean_rev_aux(t),sigma_rev_aux(t));
    pd_rev_trunc = truncate(pd_rev_parent,0,inf);
    rev_sim = random(pd_rev_trunc,NS_cf,N_t);
    
    % Compute combined signal.
    rho_sim = costs_sim - rev_sim; 
    % Compute markups by interpolating from gross auction markups.
    % Need to do several saftey checks here to avoid NaN or inf in markups.
    % Safety check to ensure that cost grids are ordered in correct 
    MU_I_ip_data = sortrows([cost_I_grid;MU_I]');
    MU_E_ip_data = sortrows([cost_E_grid;MU_E]');
    % Initialize interpolants for both bidder types.
    MU_I_ip = griddedInterpolant(MU_I_ip_data(:,1),MU_I_ip_data(:,2));
    MU_E_ip = griddedInterpolant(MU_E_ip_data(:,1),MU_E_ip_data(:,2));;
    % Interpolate markups at new net cost signals.
    MU_I_cf_sim = MU_I_ip(rho_sim(:,1));
    MU_E_cf_sim = MU_E_ip(rho_sim(:,2:end));
    
    % Compute bid.
    bids_sim = [rho_sim(:,1) + MU_I_cf_sim , rho_sim(:,2:end) + MU_E_cf_sim];
    
    % Indicate cost-efficient firm.
    [cost_min, cost_min_idx] = min(costs_sim, [], 2);
    % Indicate firm with lowest net cost signal (cost-revenue).
    [netcost_min, netcost_min_idx] = min(rho_sim, [], 2);
    % Indicate winning firm, i.e., firm with lowest cost signal.
    [bid_min, min_bid_idx] = min(bids_sim, [], 2);
    
    % What if firms simply bid their combined signal without any strategic interactions.
    % This should isolate the effect of additional noise by revenue signal?
    eff_rho_ind = (cost_min_idx==netcost_min_idx);
    eff_prob_cf_netgross_3(t,1) = mean(eff_rho_ind);
    % Complute vector whether efficient wins auction.
    eff_ind = (cost_min_idx==min_bid_idx);
    eff_prob_cf = mean(eff_ind);
    % Assign efficiency probability to container.
    eff_prob_cf_netgross_2(t,1) = eff_prob_cf; 
    end
end 

% Safety check for outliers.
eff_prob_cf_netgross_2(eff_prob_cf_netgross_2==0) = NaN;
eff_prob_cf_netgross_3(eff_prob_cf_netgross_3==0) = NaN;

% Write new log-file.
diary(project_paths('OUT_ANALYSIS','eff_prob_cf_netgross_comp.log'));
% Compute average efficiency probabilities.
fprintf('FOR NET AUCTIONS WITHOUT ANY STRATEGIC BIDDING\n\n');
fprintf('Median efficiency probability: %6.4f. \n', nanmedian(eff_prob_cf_netgross_3));
fprintf('Mean efficiency probability: %6.4f. \n\n', nanmean(eff_prob_cf_netgross_3));

% Compute average efficiency probabilities.
fprintf('FOR NET AUCTIONS WITH STRATEGIC MARKUPS AS IN GROSS\n\n');
fprintf('Median efficiency probability: %6.4f. \n', nanmedian(eff_prob_cf_netgross_2));
fprintf('Mean efficiency probability: %6.4f. \n\n', nanmean(eff_prob_cf_netgross_2));

% Write efficiency probabilities to file.
% Export data to csv file for reformatting in Python.
VarNames_cf_netgross_comp = {'cf_netgross_nonstrategic','cf_netgross_grossmu'};
eff_prob_netgross_comp = table(eff_prob_cf_netgross_3, eff_prob_cf_netgross_2, ...
			  'VariableNames', VarNames_cf_netgross_comp);
% Export efficiency data to to csv-file.
writetable(eff_prob_netgross_comp, project_paths('OUT_ANALYSIS','eff_prob_netgross_comp.csv'));

% Ratio of average markups for each line.
MU_average(:,3) = MU_average(:,2) ./ MU_average(:,1);
fprintf('FOR NET AUCTION SAMPLE PROCURED AS GROSS\n\n');
fprintf('Median markup of incumbent: %6.4f. \n', nanmedian(MU_average(:,1)));
fprintf('Median markup of entrant: %6.4f. \n', nanmedian(MU_average(:,2)));
fprintf('Median ratio of entrant over incumbent markup: %6.4f. \n', nanmedian(MU_average(:,3)));

% Export data to csv file for reformatting in Python.
VarNames_mu_netgross = {'MU_I','MU_E','ratio'};
mu_data_netgross = table(MU_average(:,1), MU_average(:,2), MU_average(:,3), ...
			  'VariableNames', VarNames_mu_netgross);
% Export efficiency data to to csv-file.
writetable(mu_data_netgross, project_paths('OUT_ANALYSIS','average_mu_netgross.csv'));
diary close;