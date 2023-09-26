%{
    Estimate entry cost for each auction based on gross auction sample cost estimates.
    from *estimate_gross_auctions.m*.

%}

clear
clc
format('short');
% Define necessary globals.
global data N_obs K

% Set seed in case any simulation is used.
rng(123456);
% SET PAPER SIZE FOR FIGURES TO AVOID LARGE MARGINS.
figuresize(15,12,'cm')
% % Prevent graphs from popping up.
% set(gcf,'Visible', 'on'); 

% Load data.
load(project_paths('OUT_ANALYSIS','grossauction_workspace'));
% Extract number of potential entrants. 
% For entry estimation, construct everything in terms of numbers of
% ENTRANTS ONLY.
N_pot = N_pot_matrix(:,end-1);
% Extract maximum number of potential entrants across auctions.
N_pot_max = max(N_pot);
% Load probability distribution of different bidder configurations.
load(project_paths('OUT_ANALYSIS','ga_entry_n_probs'));
prob_N_grid = prob_N_grid_gross;


% Safety measure to avoid "too negative" cost signals. This should not have a significant effect on the estimation below.
neg_cost_threshold = -2;

% Compute bid function parameters for all lines.
% Add third dimension to account for different number of bidders.
% Initialize empty containers.
lambda_n_pot_I = zeros(N_obs,N_pot_max);
lambda_n_pot_E = zeros(N_obs,N_pot_max);
rho_n_pot_I = zeros(N_obs,N_pot_max);
rho_n_pot_E = zeros(N_obs,N_pot_max);

% Loop over potential entrant numbers and compute bid distribution parameters for each combination.
% Create copy of entry regressors
X_entry_grid = X_orig;
for n = 1:N_pot_max
    % Update number of bidders.
    % Since this is about actual bid distributions, we have to add 1 bidder (incumbent).
    nn_trunc = min(5, n+1);
    X_entry_grid(:,end) = log(nn_trunc)./10 .* ones(N_obs,1);
    [~, lambda_vec_sim_I, lambda_vec_sim_E, rho_vec_sim_I, rho_vec_sim_E] = sim_wb_param(X_entry_grid,theta_opt);
    lambda_n_pot_I(:,n) = lambda_vec_sim_I;
    lambda_n_pot_E(:,n) = lambda_vec_sim_E;
    rho_n_pot_I(:,n) = rho_vec_sim_I;
    rho_n_pot_E(:,n) = rho_vec_sim_E;
end

% Initialize bid grid for each auction and number of bidders.
% Discretize bid distrbution in n_bid_grid grid points. 
n_bid_grid = 500;
bid_grid = zeros(N_obs, N_pot_max, n_bid_grid);
% Initialize container for expected profits for auction-n bidder combination.
E_profit_E = zeros(N_obs,N_pot_max);
% Initialize container for entrant's entry cost.
kappa_E = zeros(N_obs,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loop over all gross contracts to compute cost distributions.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for t=1:N_obs
    % Loop over all potential number of bidders (entrants only!).
    for nn=1:N_pot(t)
        % Extract bid function parameters for line t and nn entrants entering.
        lambda_I_sim = lambda_n_pot_I(t,nn);
        lambda_E_sim = lambda_n_pot_E(t,nn);
        rho_I_sim = rho_n_pot_I(t,nn);
        rho_E_sim = rho_n_pot_E(t,nn);

        % Compute bounds of bid grid (ignore 1% of extreme bids in total).
        bid_min_nt = wblinv(0.005,lambda_E_sim,rho_E_sim);
        bid_max_nt = wblinv(0.995,lambda_E_sim,rho_E_sim);
        % Construct bid grid at which to compute expected profits for a given bid.
        bid_grid_nt = linspace(bid_min_nt, bid_max_nt, n_bid_grid);
        bid_grid(t,nn,:) = bid_grid_nt;

        % Compute cost signals for each bidder type when evaluated at winning bids.
        % Compute markup term for incumbent.
        % Compute CDF and PDF for both types at bid grid.
        [CDF_I, PDF_I] = eval_bf(lambda_I_sim, rho_I_sim, bid_grid_nt, truncation_bid);
        [CDF_E, PDF_E] = eval_bf(lambda_E_sim, rho_E_sim, bid_grid_nt, truncation_bid);

        %% This is in terms of bid distributions with nn_IE including the incumbent.
        nn_IE = nn + 1;
        % Numerator of incumbent's markup term.
        G_MI = (1-CDF_E).^(nn_IE-1);
        % Denominator of entrant's markup term.
        g_MI = (nn_IE-1) .* (1-CDF_E).^(nn_IE-2) .* PDF_E;

        % Compute incumbent's markup.
        MU_I = G_MI ./ g_MI;
        % Compute incumbent's winning cost signal.
        RHO_GROSS_I = bid_grid_nt - MU_I;

        % Compute markup term for entrants.
        % Denominator of entrnat's markup term.
        % Special case for only one entrant and one incumbent.
        if nn_IE==2
            % Numerator of entrant's markup term.
            G_ME = 1-CDF_I;
            g_ME = PDF_I; % .* (1-CDF_E) .^ (nn_IE-2);
        else
            % Numerator of entrant's markup term.
            G_ME = (1-CDF_E).^(nn_IE-2) .* (1-CDF_I);
            g_ME = (nn_IE-2) .* (1-CDF_E).^(nn_IE-3) .* PDF_E  .* (1-CDF_I) + PDF_I .* (1-CDF_E) .^ (nn_IE-2);
        end

        % Compute markup for entrant.
        MU_E = (G_ME ./ g_ME)';
        MU_E(isnan(MU_E)) = 0;
        % Compute vector of entrants' costs.
        RHO_GROSS_E = bid_grid_nt' - MU_E;
        % Truncate cost draws at -2?
        RHO_GROSS_E(RHO_GROSS_E < neg_cost_threshold) = neg_cost_threshold;
        % Costs cannot be less than total amount paid for track access
        % charges?
        RHO_GROSS_E(RHO_GROSS_E < 10*data(t,18)) = 10.*data(t,18);
        MU_E_truncated = bid_grid_nt' - RHO_GROSS_E; 
        % Compute expected profits of entrant integrating over bids.
        E_profit_E(t,nn) = trapz(bid_grid_nt, PDF_E' .* MU_E_truncated .* G_ME');
    end
end

% Integrate over all potential bidder configurations.
% Entrant's entry cost is given by expected profit from entering auction.
kappa_E = sum(E_profit_E(:,1:end) .* prob_N_grid(:,1:end),2);

% Compute some summary statistics on entry costs.
kappa_E_mean = mean(kappa_E);
kappa_E_median = median(kappa_E);
kappa_E_std = std(kappa_E);
kappa_E_relative = kappa_E ./ bid_win;
str_mean = sprintf('Average entry cost for gross auction sample: %.4f (in Mio. EUR)\n', 10 * kappa_E_mean )
str_median = sprintf('Median entry cost for gross auction sample: %.4f (in Mio. EUR)\n', 10 *kappa_E_median )
str_std = sprintf('Standard deviation of entry cost for gross auction sample: %.4f (in Mio. EUR)\n', 10 *kappa_E_std )

str_mean_rel = sprintf('Average entry cost for gross auction sample relative to winning bid: %.4f (in percent)\n', 100 * mean(kappa_E_relative) )
str_median_rel = sprintf('Median entry cost for gross auction sample: %.4f (in percent)\n',100 * median(kappa_E_relative) )

% Compare winning bids to estimate entry costs.
compare_bid_kappa = [bid_win, kappa_E, bid_win - kappa_E, kappa_E_relative, N];

%% Plot some histograms to illutrate distriution of estimated entry costs.
clf;
hist(kappa_E*10, 25);
title('Distribution of entry costs (gross auction sample) - histogram')
xlabel('Entry cost estimate (in Mio. EUR)');
ylabel('Frequency');
axis([0,15,0,20]);
saveas(gcf,project_paths('OUT_FIGURES','entry_cost_gross_hist'),'pdf');

% Plot histogram of relative entry costs in net auctions.
hist(kappa_E_relative, 20);
title('Distribution of relative entry costs (gross auction sample) - histogram')
xlabel('Estimated entry costs relative to winning bid');
ylabel('Frequency');
axis([0,0.3,0,20]);
saveas(gcf,project_paths('OUT_FIGURES','entry_cost_rel_gross_hist'),'pdf');

% Split entry cost by time.
year = data(:,17);
kappa_E_early = kappa_E(year<=2004);
kappa_E_late = kappa_E(year>2004);

kappa_mean_early = mean(kappa_E_early);
kappa_median_early = median(kappa_E_early);
kappa_std_early = std(kappa_E_early);

kappa_median_late = median(kappa_E_late);
kappa_mean_late = mean(kappa_E_late);
kappa_std_late = std(kappa_E_late);

str_mean_early = sprintf('Average entry cost for early gross auction sample: %.4f (in Mio. EUR) \n', 10 * kappa_mean_early )
str_median_early = sprintf('Median entry cost for early gross auction sample: %.4f (in Mio. EUR) \n', 10 *kappa_median_early )
str_std_early = sprintf('Standard deviation of entry cost for early gross auction sample: %.4f (in Mio. EUR)\n', 10 *kappa_std_early )

str_mean_late = sprintf('Average entry cost for late gross auction sample: %.4f (in Mio. EUR) \n', 10 * kappa_mean_late )
str_median_late = sprintf('Median entry cost for late gross auction sample: %.4f (in Mio. EUR) \n', 10 *kappa_median_late )
str_std_late = sprintf('Standard deviation of entry cost for gross auction sample: %.4f (in Mio. EUR)\n', 10 *kappa_std_late )


subplot(2,1,1)
% Plot histogram of entry costs in net auctions.
hist(kappa_E_early*10, 15);
title('Distribution of entry costs (gross auction sample: 1996-2004) - histogram')
xlabel('Entry cost estimate (in Mio. EUR)');
ylabel('Frequency');
axis([0,15,0,15]);
subplot(2,1,2);
hist(kappa_E_late*10, 15);
title('Distribution of entry costs (gross auction sample: 2005-2011) - histogram')
xlabel('Entry cost estimate (in Mio. EUR)');
ylabel('Frequency');
axis([0,15,0,15]);
saveas(gcf,project_paths('OUT_FIGURES','entry_cost_gross_timesplit_hist'),'pdf');

%% Write some entry cost estimates to txt file.
str_logtitle = sprintf(strcat('Entry Cost Estimates for Gross Sample\nEstimated on:\n', date,'\n\n'));

fid = fopen(project_paths('OUT_ANALYSIS','entry_results_gross.txt'),'wt');
fprintf(fid,str_logtitle);
fprintf(fid,'1. Absolute entry cost estimates for whole gross sample:\n');
fprintf(fid,str_mean);
fprintf(fid,str_median);
fprintf(fid,str_std);
fprintf(fid,'\n\n');
fprintf(fid,'2. Entry cost estimates relative to winning bid for whole gross sample:\n');
fprintf(fid,str_mean_rel);
fprintf(fid,str_median_rel);
fprintf(fid,'\n\n');
fprintf(fid,'3. Entry cost estimates split by early/late gross sample:\n');
fprintf(fid,str_mean_early);
fprintf(fid,str_median_early);
fprintf(fid,str_std_early);
fprintf(fid,'\n');
fprintf(fid,str_mean_late);
fprintf(fid,str_median_late);
fprintf(fid,str_std_late);
fclose(fid);

% Export essential data to csv file for formatting of results in table.
entry_results = [year, N, bid_win, kappa_E];
% Save distribution of entry costs to csv-file for formatting table for paper.
dlmwrite(project_paths('OUT_ANALYSIS','entry_results_gross.csv'),entry_results);
% Save workspace for entry estimation on gross sample.
save(project_paths('OUT_ANALYSIS','grossauction_workspace_entry'));