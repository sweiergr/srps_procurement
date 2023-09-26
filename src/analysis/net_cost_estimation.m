%{
    Compute cost functions for net contracts based on hypothetical
    gross bid functions based on parameter estimates from
    *estimate_gross_auctions.m*. The computed cost distributions are used to 
    estimate alpha and revenue parameters in *net_revenue_estimation.m*.

%}

clear
clc
format('short');
% Define necessary globals.
global N_obs K

% Set seed in case any simulation is used.
rng(123456);
% Indicate what to do in terms of auxiliary plots and tests.
update_plots = 1;
check_logconcavity = 0;
% Prevent graphs from popping up.
set(gcf,'Visible', 'on'); 
% SET PAPER SIZE FOR FIGURES TO AVOID LARGE MARGINS.
figuresize(15,12,'cm')
% Set truncation bid to zero since it's not relevant.
truncation_bid = 0;
%% Load net auction workspace and gross auction parameters.
% Estimated bid function parameters from gross sample.
load(project_paths('OUT_ANALYSIS','ga_est_results'));
theta_gross = theta_opt;
clear theta_opt;
% Load net auction workspace (saved after first step on net sample).
load(project_paths('OUT_ANALYSIS','na_2step'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Compute hypothetical gross bid functions for each net auction.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define number of observations.
T = length(bid_win);
N_obs = T;
% Regressor matrix (assuming regressors are the same in gross and net sample).
X_gross_orig = X_orig;
Kg = size(X_gross_orig,2);
% Expand regressor matrix.
X_gross = repmat([repmat(db_win,1,Kg) .* X_gross_orig, repmat((1-db_win),1,Kg) .* X_gross_orig],1,2);
X_aux = X_gross(:,1:Kg) + X_gross(:,Kg+1:2*Kg);

% Compute hypothetical bid distribution parameters for each line for both incumbent and
% entrant. 
lambda_I_ng = exp(X_aux * theta_gross(1:Kg)); 
lambda_E_ng = exp(X_aux * theta_gross(Kg+1:2*Kg));
rho_I_ng = exp(X_aux * theta_gross(2*Kg+1:3*Kg));
rho_E_ng = exp(X_aux * theta_gross(3*Kg+1:4*Kg));

% Plot bid hypothetical bid functions for all lines.
if update_plots==1
    figuresize(15,12,'cm')
    plot_bid_functions(lambda_I_ng, lambda_E_ng, rho_I_ng , rho_E_ng , 'Hypothetical bid distributions for net contracts ', ...
                        'net_hypbf',[0,50]);
end                

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                
%% Simulate cost distributions based on hypothetical gross bid functions.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Initialize container for cost densities.
% At how many points to evaluate cost distribution.
kd_no_gridpoints = 2000;
% Dimesion 1: grid points for cost density: kd_no_gridpoints values.
% Dimension 2: auction number.
% Dimension 3: incumbent (1) vs. entrant (2)
% Dimension 4: value for density (1) and density grid (2).
kdens_container = zeros(kd_no_gridpoints,T,2,2);
% Container for checking whether cost densities integrate to one for each track and bidder type.
kd_int = zeros(T,2);
% Initialize containers for counting how many negative costs.
% First row for incumbent, second row for entrant.
neg_costs = zeros(2,T);

%% Evaluate cdf and pdf of incumbent bid function at winning bid.
[CDF_I, PDF_I] = eval_bf(lambda_I_ng,rho_I_ng, bid_win);
% Evaluate cdf and pdf of entrant bid function at winning bid.
[CDF_E, PDF_E] = eval_bf(lambda_E_ng,rho_E_ng, bid_win);


%% Prepare FOSD test and set its parameters.
% Number of grid points to use for FOSD test.
N_a = 10;
% Test for FOSD on larger range: 0 to 20.
a_grid_full = linspace(0,20,N_a);
% Test for FOSD only on lower tail: 0 to 6. (we may need to update what makes sense here or make this line-specific!)
a_grid_lower = linspace(2,6,N_a);
% Collect values of empirical CDFs for costs of incumbent and entrants (at grid points).    
eCDF_cost_full = zeros(N_obs,N_a,2);
eCDF_cost_lower = zeros(N_obs,N_a,2);

% Stack variance of empirical CDF of incumbent, entrant and difference in third dimension.
cov_FI_FE_full = zeros(N_obs,N_a);
var_a_fosd_full = zeros(N_obs,N_a,4);
cov_FI_FE_lower = zeros(N_obs,N_a);
var_a_fosd_lower = zeros(N_obs,N_a,4);
% Container for test statistic at grid points.
T_grid_full = zeros(N_obs,N_a);
T_grid_lower = zeros(N_obs,N_a);
% Determine critical values manually (they depend on N_a and significance
% level alpha): which alpha-value would we like? (add legend for significance levels)
% Critical value for singificance level 1%
m_kalpha = 3.29;
% Critical value for singificance level 5%
m_kalpha = 2.8;
% Critical value for singificance level 10%
%m_kalpha = 2.560;
% Container vector for rejection decisions.
% H0: Cost distributions are equal.
% H1: Entrant cost distriubtion dominates incumbent's.
% First column: test full distribution.
% Second column: test only lower range.
reject_equality = zeros(N_obs,2);

%% Prepare cost distribution analysis.
% Set quantile grid at which to compare incumbent and entrant cost function.
quantile_grid = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 0.9, 0.95,0.99];
% Initialize container for cost characteristics for mean and various quantiles.
% Incumbent in first level, entrant in second.
cost_quantiles = zeros(N_obs,size(quantile_grid,2)+1,2);
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% IMPORTANT TUNING PARAMETERS WHEN ESTIMATING COSTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Here it has a direct effect on the second step net estimation.
% These "tuning parameters" are mostly relevant for outlier observations that are excluded from the final sample.
% Set negative cost threshold: this is the lowest negative cost that we allow.
% Have these thresholds line-specific.
neg_cost_threshold = -2 + 0 .* bid_win; 
% Potentially truncate costs from above. Not sure this is needed.
pos_cost_threshold = 20 + 3*bid_win;; 

% The following parameter probably do not have a big effect.
% Set number of Kernel density grid points.
N_gridpoints = kd_no_gridpoints;
% Set bandwidth for kernel density of cost pdf 
kd_bandwidth = 0.5;
% Set number of simulated auctions.
NS_auctions = 10000;
% Construct container for markup components (useful for checking where markups become weird).
G_I = zeros(NS_auctions, N_obs);
g_I = zeros(NS_auctions, N_obs);
G_E = zeros(NS_auctions, N_obs);
g_E = zeros(NS_auctions, N_obs);

% Compute bid grid.
n_bid_grid = 5000;
lambda_I_grid = repmat(lambda_I_ng,1,n_bid_grid);
lambda_E_grid = repmat(lambda_E_ng,1,n_bid_grid);
rho_I_grid = repmat(rho_I_ng,1,n_bid_grid);
rho_E_grid = repmat(rho_E_ng,1,n_bid_grid);
% Evaluate Weibull CDF between 0 and 150.
bid_grid = repmat(linspace(0,150,n_bid_grid),N_obs,1);
% Compute estimated bid distributions manually.
% At some point we can switch this to using MATLAB's Weibull CDF function.
bid_distribution_I = ones(N_obs,n_bid_grid) - exp( - (bid_grid ./ lambda_I_grid) .^ (rho_I_grid) ); 
bid_distribution_E = ones(N_obs,n_bid_grid) - exp( - (bid_grid ./ lambda_E_grid) .^ (rho_E_grid) ); 
bid_density_I = exp( - (bid_grid ./ lambda_I_grid)) .^ (rho_I_grid-1) .* (bid_grid ./ lambda_I_grid);
bid_density_E = exp( - (bid_grid ./ lambda_E_grid)) .^ (rho_E_grid-1) .* (bid_grid ./ lambda_E_grid);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% END OF TUNING PARAMETERS              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Construct container for markup components (useful for checking where markups become weird).
G_I = zeros(NS_auctions, N_obs);
g_I = zeros(NS_auctions, N_obs);
G_E = zeros(NS_auctions, N_obs);
g_E = zeros(NS_auctions, N_obs);
% Containers for winning bid signals.
g_I_win = zeros(N_obs,1);
G_I_win = zeros(N_obs,1);
g_E_win = zeros(N_obs,1);
G_E_win = zeros(N_obs,1);
% Container for winner's siganl for each type.
RHO_GROSS = zeros(N_obs,2);
% Containers for winning bid signals.
g_I_bf = zeros(N_obs,n_bid_grid);
G_I_bf = zeros(N_obs,n_bid_grid);
g_E_bf = zeros(N_obs,n_bid_grid);
G_E_bf = zeros(N_obs,n_bid_grid);
signal_I_bf = zeros(N_obs,n_bid_grid);
signal_E_bf = zeros(N_obs,n_bid_grid);

% Simple Kolmogorov Smirnov Test for equality of cost distributions.
ks_test_reject= zeros(N_obs,1);
% Add containers for complete cost density estimates and CDFs.
ksdens_I = zeros(N_obs,n_bid_grid);
ksdens_I_grid = zeros(N_obs,n_bid_grid);
ksCDF_I = zeros(N_obs,n_bid_grid);
ksCDF_I_grid = zeros(N_obs,n_bid_grid);
ksdens_E = zeros(N_obs,n_bid_grid);
ksdens_E_grid = zeros(N_obs,n_bid_grid);
ksCDF_E = zeros(N_obs,n_bid_grid);
ksCDF_E_grid = zeros(N_obs,n_bid_grid);

%% Initialize containers for various markup statistics.
% First two columns indicate statistics  for incumbent (mean and median), last two columns are statistics for entrant. 
MU_abs = zeros(N_obs,4); % absolute value of markup considering full markup distribution.
MU_rel = zeros(N_obs,4); % value of markup relativ to winning bid considering full markup distribution.
MU_abs_clean = zeros(N_obs,4); % absolute value of markup considering distribution of markups but with tails truncated.
MU_rel_clean = zeros(N_obs,4); % value of markup relatve to winning bid considering distribution of markups but with tails truncated.

% Cost distribution for each line.
for t=1:size(X_orig,1);
    % Extract bid function parameters for line t.
    lambda_I_sim = lambda_I_ng(t,1);
    lambda_E_sim = lambda_E_ng(t,1);
    rho_I_sim = rho_I_ng(t,1);
    rho_E_sim = rho_E_ng(t,1);
    % Simulate bids from estimated bid distributions.
    % Resetting the seed for each line is not really necessary and doesn't affect the results much.
    rng(123456)
    bid_draw_I = wblrnd(lambda_I_sim,rho_I_sim,[NS_auctions,1]);
    bid_draw_E = wblrnd(lambda_E_sim,rho_E_sim,[NS_auctions,1]);

    % Trim draws from below at total track access costs: Never bid less than track access charges. Makes sense when gross auctions are procured.
    bid_draw_I(bid_draw_I < data(t,22)) = 1.0 .* data(t,22);
    bid_draw_E(bid_draw_E < data(t,22)) = 1.0 .* data(t,22);
    % Compute markup terms for incumbent.
    % Compute CDF and PDF for both types at incumbent's simulated bid.
    [CDF_I, PDF_I] = eval_bf(lambda_I_sim,rho_I_sim,bid_draw_I,truncation_bid);
    [CDF_E, PDF_E] = eval_bf(lambda_E_sim,rho_E_sim,bid_draw_I,truncation_bid);
    % Numerator of incumbent's markup term.
    G_MI = (1-CDF_E).^(N(t)-1);
    % Denominator of incumbent's markup term.
    g_MI = (N(t)-1) .* (1-CDF_E).^(N(t)-2) .* PDF_E;
    % Write numerator and denominator into container.
    g_I(:,t) = g_MI;
    G_I(:,t) = G_MI;
    % Compute incumbent's markup.
    MU_I = G_MI ./ g_MI;
    MU_I(G_MI==0) = 0;
    % Compute vector of simulated incumbent's costs.
    cost_I_grid = bid_draw_I - MU_I;
    
    % DEBUGGING/EXPERIMENTATION: INCREASE COSTS OF INCUMBENT!
    % cost_I_grid = cost_I_grid + 5;
    
    %% Check how many cost draws are negative.
    sprintf('Fraction of negative cost draws for incumbent:')
    neg_cost_I = sum(cost_I_grid<0) ./ NS_auctions
    neg_costs(1,t) = neg_cost_I;

    % Compute markup terms for entrant.
    % Compute CDF and PDF for both types evaluated at entrants' bid.
    [CDF_I, PDF_I] = eval_bf(lambda_I_sim,rho_I_sim,bid_draw_E,truncation_bid);
    [CDF_E, PDF_E] = eval_bf(lambda_E_sim,rho_E_sim,bid_draw_E,truncation_bid);
    % Numerator of entrants' markup term.
    G_ME = (1-CDF_E).^(N(t)-2) .* (1-CDF_I);
    % Denominator of entrants' markup term.
    g_ME = (N(t)-2) .* (1-CDF_E).^(N(t)-3) .* PDF_E  .* (1-CDF_I) + PDF_I .* (1-CDF_E) .^ (N(t)-2);
    % Write numerator and denominator into container.
    G_E(:,t) = G_ME;
    g_E(:,t) = g_ME;
    % Compute markup term for entrant.
    MU_E = G_ME ./ g_ME;
    MU_E(G_ME==0) = 0;

    % Compute vector of simulated entrants' costs.
    cost_E_grid = bid_draw_E - MU_E;
    % PURELY FOR DEBUGGING AND EXPERIMENTAL PURPOSES:
    % cost_I_grid = cost_E_grid;
    % Check how many cost draws are negative.
    sprintf('Fraction of negative cost draws for entrant:')
    neg_cost_E = sum(cost_E_grid<0) ./ NS_auctions
    neg_costs(2,t) = neg_cost_E;

    % Compute and plot distribution of markups.
    MU_E_plot = MU_E(MU_E<prctile(MU_E,90) & MU_E>prctile(MU_E,10));
    MU_I_plot = MU_I(MU_I<prctile(MU_I,90) & MU_I>prctile(MU_I,10));
    fprintf('Median markup for entrant and incumbent in gross auction %d: %6.4f and %6.4f\n',t, nanmedian(MU_I), nanmedian(MU_E));
    fprintf('Median markup for entrant and incumbent in gross auction %d (outliers cleaned): %6.4f and %6.4f\n',t, nanmedian(MU_I_plot), nanmedian(MU_E_plot));
    fprintf('Average markup for entrant and incumbent in gross auction %d: %6.4f and %6.4f\n',t, nanmean(MU_I), nanmean(MU_E));
    fprintf('Average markup for entrant and incumbent in gross auction %d (outliers cleaned): %6.4f and %6.4f\n',t, nanmean(MU_I_plot), nanmean(MU_E_plot));
    
    %% Compute smoothed markup distribution for both bidder types.
    try 
    % Markups for incumbent.
    [mu_density_I,mu_grid_I] = ksdensity(MU_I_plot);
    catch me
        fprintf('Weird markup distribution for incumbent in gross auction %d. Investigate!\n',t);
        mu_density_I = zeros(100,1);
        mu_grid_I = zeros(100,1);
    end
    try
        % Markups for entrant.
        [mu_density_E,mu_grid_E] = ksdensity(MU_E_plot);
    catch me
        fprintf('Weird markup distribution for entrant in gross auction %d. Investigate!\n',t);
        mu_density_E = zeros(100,1);
        mu_grid_E = zeros(100,1);
    end
    % Plot smoothed markup distribution for both bidder types.
    subplot(1,1,1)
    plot(mu_grid_I,mu_density_I,mu_grid_E,mu_density_E);
    graphtitle = sprintf('Comparison of MU distribution for net contract %d procured gross' ,t);
    title(graphtitle);
    legend('Incumbent', 'Entrant')
    filename = sprintf(regexprep(strcat(project_paths('OUT_FIGURES','mu_comp_cfnetgross'),num2str(t)),'\','\\\'));
    saveas(gcf,filename,'pdf');

    %% Compute markup statistics for auction t.
    % Absolute value of markup not cleaning for outliers in markup
    % distribution.
    MU_abs(t,1) = nanmean(MU_I);
    MU_abs(t,2) = nanmedian(MU_I);
    MU_abs(t,3) = nanmean(MU_E);
    MU_abs(t,4) = nanmedian(MU_E);
    % Value of markup relative to winning bid not cleaning for outliers in
    % markup distribution.
    MU_rel(t,1) = nanmean(MU_I ./ bid_win(t));
    MU_rel(t,2) = nanmedian(MU_I ./ bid_win(t));
    MU_rel(t,3) = nanmean(MU_E ./ bid_win(t));
    MU_rel(t,4) = nanmedian(MU_E ./ bid_win(t));
    % Absolute value of markup cleaning for outliers in markup distribution.
    MU_abs_clean(t,1) = nanmean(MU_I_plot);
    MU_abs_clean(t,2) = nanmedian(MU_I_plot);
    MU_abs_clean(t,3) = nanmean(MU_E_plot);
    MU_abs_clean(t,4) = nanmedian(MU_E_plot);
    % Value of markup relative to winning bid cleaning for outliers in markup distribution.
    MU_rel_clean(t,1) = nanmean(MU_I_plot ./ bid_win(t));
    MU_rel_clean(t,2) = nanmedian(MU_I_plot ./ bid_win(t));
    MU_rel_clean(t,3) = nanmean(MU_E_plot ./ bid_win(t));
    MU_rel_clean(t,4) = nanmedian(MU_E_plot ./ bid_win(t));
    
    %% Truncate costs.
    % Negative costs (probably most important for us, but not super relevant in final sample)
    cost_I_grid(cost_I_grid<neg_cost_threshold(t)) = neg_cost_threshold(t);
    cost_E_grid(cost_E_grid<neg_cost_threshold(t)) = neg_cost_threshold(t);
    % Positive costs (probably not relevant for us, but not super relevant in final sample)
    cost_I_grid(cost_I_grid>pos_cost_threshold(t)) = pos_cost_threshold(t);
    cost_E_grid(cost_E_grid>pos_cost_threshold(t)) = pos_cost_threshold(t);
    % Update x-axis scale for costs.
    x_axis_cost = [zeros(T,1) 16 .* ones(T,1)];


    %% Plot and smooth cost estimates.
        % Plot empirical CDF of incument's cost in first subplot.
        subplot(2,1,1)
        cdfplot(cost_I_grid);
        title_str = sprintf(strcat('Estimated cost CDF of incumbent for net contract ',num2str(t)));
        title(title_str,'Interpreter','latex')
        % POTENTIALLY NEED TO ADJUST SCALING HERE!
        axis([x_axis_cost(t,1) x_axis_cost(t,2) 0 1])
        % Plot empirical CDF of entrant's cost in second subplot.
        subplot(2,1,2)
        cdfplot(cost_E_grid);
        title_str = sprintf(strcat('Estimated cost CDF of entrant for net contract ',num2str(t)));
        title(title_str,'Interpreter','latex')
        axis([x_axis_cost(t,1) x_axis_cost(t,2) 0 1])
        filename = sprintf(regexprep(strcat(project_paths('OUT_FIGURES','cdfcostnet'),num2str(t)),'\','\\\'));
        saveas(gcf,filename,'pdf');
        
        %% Kernel densities of costs.
        % Compute kernel density of incumbent's cost in first subplot.
        subplot(2,1,1)
        % Compute cost density at bid grid for computation of bid
        % functions.
        [ksdens_I(t,:),ksdens_I_grid(t,:)] = ksdensity(cost_I_grid,bid_grid(t,:),'bandwidth',kd_bandwidth,'NumPoints', N_gridpoints);
        % Compute Kernel CDF for incumbent cost distribution.
        [ksCDF_I(t,:),ksCDF_I_grid(t,:)] = ksdensity(cost_I_grid,bid_grid(t,:), 'bandwidth',kd_bandwidth,'NumPoints', N_gridpoints_kd ,'Function','cdf');
        % Compute cost density at flexible bid grid for saving to
        % container.
        [kdens_I,kdens_I_grid] = ksdensity(cost_I_grid,'bandwidth',kd_bandwidth,'NumPoints', N_gridpoints);
        % Compute expected cost for incumbent by integrating numerically over kerndel density.
        cost_quantiles(t,1,1) = trapz(ksdens_I_grid(t,:),ksdens_I_grid(t,:) .* ksdens_I(t,:));
        % Compute cost quantiles of incumbent's cost distribution.
        cq_I = ksdensity(cost_I_grid, quantile_grid, 'bandwidth',kd_bandwidth,'NumPoints', N_gridpoints, 'function','icdf');
        cost_quantiles(t,2:end,1) = cq_I;
        % Extract CDF values for FOSD test at grid points.
        eCDF_cost(t,:,1) = ksdensity(cost_I_grid, a_grid_full, 'bandwidth',kd_bandwidth,'NumPoints', N_gridpoints, 'function','cdf');
       
        % Write densities for incumbent to container matrix.
        % This is the cost information that we integrate over in the likelihood of the second step.
        kdens_container(:,t,1,1) = kdens_I;
        kdens_container(:,t,1,2) = kdens_I_grid;
        % Sanity check: Integrate over cost Kernel density.
        kd_int(t,1) = trapz(kdens_I_grid,kdens_I);
                
        % Plot Kernel density for incumbent.        
        plot(kdens_I_grid, kdens_I);
        title_str = sprintf(strcat('Kernel density of incumbent cost for net contract ',num2str(t)));
        title(title_str,'Interpreter','latex')
        axis([x_axis_cost(t,1) x_axis_cost(t,2) 0 0.75])

        % Do the same thing for entrant cost distribution.
        subplot(2,1,2)
        % Compute kernel density for entrants' cost.
        % Cost density evaluated at bid grid for computation of bid
        % function.
        [ksdens_E(t,:),ksdens_E_grid(t,:)] = ksdensity(cost_E_grid,bid_grid(t,:),'bandwidth',kd_bandwidth,'NumPoints', N_gridpoints);
        % Compute Kernel CDF for incumbent cost distribution.
        [ksCDF_E(t,:),ksCDF_E_grid(t,:)] = ksdensity(cost_E_grid,bid_grid(t,:), 'bandwidth',kd_bandwidth,'NumPoints', N_gridpoints_kd ,'Function','cdf');
        % At flexible bid grid.
        [kdens_E,kdens_E_grid] = ksdensity(cost_E_grid,'bandwidth',kd_bandwidth,'NumPoints', N_gridpoints);
        % Compute expected cost for entrant.
        cost_quantiles(t,1,2) = trapz(ksdens_E_grid(t,:),ksdens_E_grid(t,:) .* ksdens_E(t,:));
        % Compute cost quantiles of entrant's cost distribution.
        cq_E = ksdensity(cost_E_grid, quantile_grid, 'bandwidth',kd_bandwidth,'NumPoints', N_gridpoints, 'function','icdf');
        cost_quantiles(t,2:end,2) = cq_E;
        % Extract CDF values for FOSD test.
        eCDF_cost(t,:,2) = ksdensity(cost_E_grid, a_grid_full, 'bandwidth',kd_bandwidth,'NumPoints', N_gridpoints, 'function','cdf');

        % Write densities of entrants' cost to container matrix.
        kdens_container(:,t,2,1) = kdens_E;
        kdens_container(:,t,2,2) = kdens_E_grid;

        % Sanity check: Integrate over entrants' Kernel density 
        kd_int(t,2) = trapz(kdens_E_grid,kdens_E);
        % Check for integral at bid grid points.
        %kd_int(t,2) = trapz(ksdens_E_grid,ksdens_E);

        % Plot kernsel density for entrant.
        plot(kdens_E_grid, kdens_E);
        title_str = sprintf(strcat('Kernel density of entrant cost for net contract ',num2str(t)));
        title(title_str,'Interpreter','latex')
        axis([x_axis_cost(t,1) x_axis_cost(t,2) 0 0.75])
        filename = sprintf(regexprep(strcat(project_paths('OUT_FIGURES','kdensitycostnet'),num2str(t)),'\','\\\'));
        saveas(gcf,filename,'pdf');
        
        % Plot bid function and cost distribution in one graph a la ALS page 244.
        % Computation of bid function: This requires solving system of
        % bidding FOCs at prespecified bid_grid.
        % Compute markup term for incumbent.
        % Compute CDF and PDF for both types at winning bid.
        [CDF_I_win, PDF_I_win] = eval_bf(lambda_I_sim,rho_I_sim,bid_grid(t,:),truncation_bid);
        [CDF_E_win, PDF_E_win] = eval_bf(lambda_E_sim,rho_E_sim,bid_grid(t,:),truncation_bid);

        % Numerator of incumbent's markup term.
        G_MI_bf = (1-CDF_E_win).^(N(t)-1);
        % Denominator of entrant's markup term.
        g_MI_bf = (N(t)-1) .* (1-CDF_E_win).^(N(t)-2) .* PDF_E_win;
        % Write numerator and denominator into container.
        g_I_bf(t,:) = g_MI_bf;
        G_I_bf(t,:) = G_MI_bf;
        % Compute incumbent's markup.
        MU_I_bf = G_MI_bf ./ g_MI_bf;
        % Compute incumbent's winning cost signal.
        signal_I_bf_aux = bid_grid(t,:) - MU_I_bf;
        signal_I_bf_aux(G_I_bf(t,:)==0) = bid_grid(t,G_I_bf(t,:)==0);
        signal_I_bf(t,:) = signal_I_bf_aux;
        % Compute markup term for entrants.
        % Numerator of entrant's markup term.
        G_ME_bf = (1-CDF_E_win).^(N(t)-2) .* (1-CDF_I_win);
        % Denominator of entrnat's markup term.
        g_ME_bf = (N(t)-2) .* (1-CDF_E_win).^(N(t)-3) .* PDF_E_win  .* (1-CDF_I_win) + PDF_I_win .* (1-CDF_E_win) .^ (N(t)-2);
        % Write numerator and denominator into container.
        G_E_bf(t,:) = G_ME_bf;
        g_E_bf(t,:) = g_ME_bf;
        % Compute markup for entrant.
        MU_E_bf = G_ME_bf ./ g_ME_bf;
        % Compute vector of entrants' costs.
        signal_E_bf_aux = bid_grid(t,:) - MU_E_bf;
        signal_E_bf_aux(G_E_bf(t,:)==0) = bid_grid(t,G_E_bf(t,:)==0);
        signal_E_bf(t,:) = signal_E_bf_aux;
        % Plot bid function and cost distribution for incumbent.
        subplot(2,1,1)
        yyaxis left;
        plot(signal_I_bf(t,:),bid_grid(t,:));
        ylabel('Bid (in 10 Mio EUR)','Interpreter','latex')
        title_str = sprintf(strcat('Gross bid function and cost distribution (incumbent) for net contract  ',num2str(t)));
        title(title_str,'Interpreter','latex')
        axis([bid_grid(t,1) bid_grid(t,n_bid_grid./5) 0 bid_grid(t,600)])
        yyaxis right;
        plot(bid_grid(t,:), ksdens_I(t,:));
        ylabel('Cost density','Interpreter','latex')
        xlabel('Cost Signal (in 10 Mio EUR)','Interpreter','latex')

        % Plot bid function and cost distribution for entrant.
        subplot(2,1,2)
        yyaxis left;
        plot(signal_E_bf(t,:),bid_grid(t,:));
        ylabel('Bid (in 10 Mio EUR)','Interpreter','latex')
        title_str = sprintf(strcat('Gross bid function and cost distribution (entrant) for net contract  ',num2str(t)));
        title(title_str,'Interpreter','latex')
        axis([bid_grid(t,1) bid_grid(t,n_bid_grid./5) 0 bid_grid(t,600)])
        yyaxis right;
        plot(bid_grid(t,:), ksdens_E(t,:));
        ylabel('Cost density','Interpreter','latex')
        xlabel('Cost Signal (in 10 Mio EUR)','Interpreter','latex')
        filename = sprintf(regexprep(strcat(project_paths('OUT_FIGURES','bfkd_net'),num2str(t)),'\','\\\'));
        saveas(gcf,filename,'pdf');
        
        % Verify that cost distribution is log-concave.
        if check_logconcavity==1
            % Plot log of kernel density of incument's and entrants' cost.
            subplot(2,1,1)
            [kdens, ksdens_grid] = ksdensity(cost_I_grid,'bandwidth',0.75,'NumPoints', N_gridpoints);
            log_kdens = log(kdens);
            plot(ksdens_grid,log_kdens)
            title_str = sprintf(strcat('Density of incumbent log-costs for net contract ',num2str(t)));
            title(title_str,'Interpreter','latex')
            %axis([0 18 0 1])
            subplot(2,1,2)
            [kdens, ksdens_grid] = ksdensity(cost_E_grid,'bandwidth',0.75,'NumPoints', N_gridpoints);
            log_kdens = log(kdens);
            plot(ksdens_grid,log_kdens);
            title_str = sprintf(strcat('Density of entrant log-costs for net contract ',num2str(t)));
            title(title_str,'Interpreter','latex')
           % axis([0 18 0 1])
            filename = sprintf(regexprep(strcat(project_paths('OUT_FIGURES','logdensitycostnet'),num2str(t)),'\','\\\'));
            saveas(gcf,filename,'pdf');
        end

        %% Compute Davidson-Duclos FOSD test.
        % Define grid for two tests of FOSD.
        % Test for FOSD on full range: 0 to 90%-percentile of bid function.
        c_full_max = max(cost_quantiles(t,8,1),cost_quantiles(t,8,2));
        c_lower_max = max(cost_quantiles(t,3,1),cost_quantiles(t,3,2));
        a_grid_full = linspace(0,10*bid_win(t),N_a);
        % Test for FOSD only on lower tail: 0 to 6. We may need to update what makes sense here or make this line-specific...
        a_grid_lower = linspace(0,3*bid_win(t),N_a);
        
        % For testing full distribution.
        % Extract CDF values for FOSD test at grid points.
        eCDF_cost_full(t,:,1) = ksdensity(cost_I_grid, a_grid_full, 'bandwidth',kd_bandwidth,'NumPoints', N_gridpoints_kd, 'function','cdf');
        eCDF_cost_full(t,:,2) = ksdensity(cost_E_grid, a_grid_full, 'bandwidth',kd_bandwidth,'NumPoints', N_gridpoints_kd, 'function','cdf');
        % Compute variance of empirical CDF for FOSD test.
        var_a_fosd_full(t,:,1) = (eCDF_cost_full(t,:,1) - eCDF_cost_full(t,:,1).^2) ./ NS_auctions;
        var_a_fosd_full(t,:,2) = (eCDF_cost_full(t,:,2) - eCDF_cost_full(t,:,2).^2) ./ NS_auctions;
        % Covariance of F_I and F_E.
        cov_FI_FE_full(t,:) = (1 - eCDF_cost_full(t,:,1) .* eCDF_cost_full(t,:,2)) ./ NS_auctions;
        % Compute variance.
        var_a_fosd_full(t,:,3) = var_a_fosd_full(t,:,1) + var_a_fosd_full(t,:,2);
        % Alternative variance formula from Davidson/Duclos.
        var_a_fosd_full(t,:,4) = - (eCDF_cost_full(t,:,1) - eCDF_cost_full(t,:,2)) .*  (eCDF_cost_full(t,:,2) - eCDF_cost_full(t,:,1)) ./ NS_auctions;
        % Compute test statistic.
        T_grid_full(t,:) = (eCDF_cost_full(t,:,2) - eCDF_cost_full(t,:,1)) ./ sqrt(var_a_fosd_full(t,:,3));
        % Compute rejection decision.
        reject_equality(t,1) = max(-T_grid_full(t,:) > m_kalpha) .* min(T_grid_full(t,:) < m_kalpha);
        
        % For testing lower part of distribution.
        % Extract CDF values for FOSD test at grid points.
        eCDF_cost_lower(t,:,1) = ksdensity(cost_I_grid, a_grid_lower, 'bandwidth',kd_bandwidth,'NumPoints', N_gridpoints_kd, 'function','cdf');
        eCDF_cost_lower(t,:,2) = ksdensity(cost_E_grid, a_grid_lower, 'bandwidth',kd_bandwidth,'NumPoints', N_gridpoints_kd, 'function','cdf');
        % Compute variance of empirical CDF for FOSD test.
        var_a_fosd_lower(t,:,1) = (eCDF_cost_lower(t,:,1) - eCDF_cost_lower(t,:,1).^2) ./ NS_auctions;
        var_a_fosd_lower(t,:,2) = (eCDF_cost_lower(t,:,2) - eCDF_cost_lower(t,:,2).^2) ./ NS_auctions;
        % Covariance of F_I and F_E.
        cov_FI_FE_lower(t,:) = (1 - eCDF_cost_lower(t,:,1) .* eCDF_cost_lower(t,:,2)) ./ NS_auctions;
        % Compute variance.
        var_a_fosd_lower(t,:,3) = var_a_fosd_lower(t,:,1) + var_a_fosd_lower(t,:,2);
        % Alternative variance formula from Davidson/Duclos.
        var_a_fosd_lower(t,:,4) = - (eCDF_cost_lower(t,:,1) - eCDF_cost_lower(t,:,2)) .*  (eCDF_cost_lower(t,:,2) - eCDF_cost_lower(t,:,1)) ./ NS_auctions;
        % Compute test statistic.
        T_grid_lower(t,:) = (eCDF_cost_lower(t,:,2) - eCDF_cost_lower(t,:,1)) ./ sqrt(var_a_fosd_lower(t,:,3));
        % Compute rejection decision.
        reject_equality(t,2) = max(-T_grid_lower(t,:) > m_kalpha) .* min(T_grid_lower(t,:) < m_kalpha);
        % Run Kolmogorov-Smirnov Test.
        ks_test_reject(t,1) = kstest2(cost_I_grid,cost_E_grid);
end

%% Analyze mean markup components for each contract. This helps in debugging code and detecting outlier observations.
g_I_mean = mean(g_I);
G_I_mean = mean(G_I);
g_E_mean = mean(g_E);
G_E_mean = mean(G_E);
% Check for implausible cost estimates.
cost_support_I = squeeze(kdens_container(:,:,1,2));
cost_support_E = squeeze(kdens_container(:,:,2,2));

%% Post-estimation analysis of cost estimates.
% Incumbent in first column, entrant in second column.
clf('reset')
% SET PAPER SIZE FOR FIGURES TO AVOID LARGE MARGINS.
figuresize(15,12,'cm')
% Median costs for each contract and bidder type.
% Incumbent in first column, entrant in second column.
median_costs = squeeze(cost_quantiles(:,7,:));
rel_cost_adv_med = (median_costs(:,2) - median_costs(:,1)) ./ abs(median_costs(:,1)) - 1;
rel_cost_adv_med(rel_cost_adv_med>10) = 10;
subplot(2,1,1)
hist(rel_cost_adv_med)
title('Relative cost advantage median of incumbent - histogram','Interpreter','latex')
mean_costs = squeeze(cost_quantiles(:,1,:));
rel_cost_adv = (mean_costs(:,2) - mean_costs(:,1)) ./ abs(mean_costs(:,1)) - 1;
rel_cost_adv(rel_cost_adv>10) = 10;
subplot(2,1,2)
hist(rel_cost_adv)
title('Relative cost advantage mean of incumbent - histogram','Interpreter','latex')
saveas(gcf,project_paths('OUT_FIGURES','rca_net'),'pdf');

% Save median and mean cost data to join with gross sample.
median_costs_net = median_costs;
mean_costs_net = mean_costs;
rel_cost_adv_med_net = rel_cost_adv_med;
rel_cost_adv_net = rel_cost_adv;
save(project_paths('OUT_ANALYSIS','rca_data_net.mat'),'median_costs_net', 'mean_costs_net', 'rel_cost_adv_med_net', 'rel_cost_adv_net');

% Compute relative cost advantage of incumbent vs entrant.
mean_cost_adv_rel = ((mean_costs(:,2)-mean_costs(:,1)) ./mean_costs(:,1) - 1);
median_cost_adv_rel = ((median_costs(:,2)-median_costs(:,1)) ./median_costs(:,1) - 1);
% When reporting these results we might have to adjust/drop outliers.
% Truncate negative costs at negative cost threshold.
outlier_indicator_below = max((mean_costs(:,1) < neg_cost_threshold),(mean_costs(:,2) < neg_cost_threshold));
% Truncate positive costs at upper threshold.
outlier_indicator_above = max((mean_costs(:,1) > pos_cost_threshold),(mean_costs(:,2) > pos_cost_threshold));
% Construct compound outlier indicator.
outlier_indicator = max(outlier_indicator_below,outlier_indicator_above);
cost_quantiles_clean = (cost_quantiles(outlier_indicator==0,:,:));
avg_cost_quantiles = mean(cost_quantiles_clean);
% Compute difference in quantiles between incumbent and entrant.
avg_cq_diff = avg_cost_quantiles(:,:,1) - avg_cost_quantiles(:,:,2);

% Incumbent probably has advantage on median and average costs.
med_cost_advantage = -mean(avg_cq_diff(:,7));
mean_cost_advantage = -mean(avg_cq_diff(:,1));
med_cost_avg = mean(cost_quantiles_clean(:,7,:));
mean_cost_avg = mean(cost_quantiles_clean(:,1,:));
% Save cost data for net auction workspace.
cq_clean_net = cost_quantiles_clean;
save(project_paths('OUT_ANALYSIS','cq_clean_net.mat'));

% Sum of lines for which we can reject equality of distributions in favor
% of entrant stochastically dominating incumbent.
sum_reject_Fcost_equality = sum(reject_equality)
share_reject_Fcost_equality = sum_reject_Fcost_equality ./ size(reject_equality,1);

%% Write some entry cost estimates to txt file.
str_logtitle = sprintf(strcat('Summary statistics for FOSD test of equality of cost distribution in net auctions\nEstimated on:\n', date,'\n\n'));
fid = fopen(project_paths('OUT_ANALYSIS','fosd_test_net.txt'),'wt');
fprintf(fid,str_logtitle);
fprintf(fid,':SUMMARY STATISTICS WHEN TESTING FOR FOSD OF COST DISTRIBUTIONS IN NET AUCTION SAMPLE:\n')
fprintf(fid,'Reject equality of entire distribution in favor of incumbent having better cost distribution on %d net auction lines.\n ', sum_reject_Fcost_equality(1));
fprintf(fid,'Reject equality of lower part of distribution in favor of incumbent having better cost distribution on %d net auction lines.\n ', sum_reject_Fcost_equality(2));
fprintf(fid,'\nEXPRESSED IN SHARE OF TOTAL NUMBER OF LINES IN NET SAMPLE:\n')
fprintf(fid,'Reject equality of entire distribution in favor of incumbent having better cost distribution on %6.4f percent of net auction lines.\n ', 100 .* share_reject_Fcost_equality(1));
fprintf(fid,'Reject equality of lower part of distribution in favor of incumbent having better cost distribution on %6.4f percent of net auction lines.\n ', 100 .* share_reject_Fcost_equality(2));
fclose(fid);

% Diagonistics: Compute cost range for incumbent anentrant.
cost_range = [cost_I_grid(1,:);cost_I_grid(end,:);cost_E_grid(1,:);cost_E_grid(end,:)];
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% END OF FIRST PART (COMPUTING COST DISTRIBUTIONS FOR NET LINES) %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Plot summary statistics of markup distributions.
% Plot histogram.
subplot(2,1,1)
hist(MU_abs_clean(:,[1,3]));
legend('Incumbent','Entrant');
title('Histogram of mean (absolute) markups');
subplot(2,1,2)
hist(MU_rel_clean(:,[1,3]));
legend('Incumbent','Entrant');
title('Histogram of mean markups (relative to winning bid)');
saveas(gcf,project_paths('OUT_FIGURES','hist_mean_mu_cfnetgross'),'pdf');

subplot(2,1,1)
hist(MU_abs_clean(:,[2,4]));
legend('Incumbent','Entrant');
title('Histogram of median (absolute) markups');
subplot(2,1,2)
hist(MU_rel_clean(:,[2,4]));
legend('Incumbent','Entrant');
title('Histogram of median markups (relative to winning bid)');
saveas(gcf,project_paths('OUT_FIGURES','hist_median_mu_cfnetgross'),'pdf');

% Write markup statistic to table.
% Two files for absolute and relative markup statistics.
mu_stats_labels = {'I_mean','I_median','E_mean','E_median'};
mu_table_abs = table(MU_abs(:,1),MU_abs(:,2),MU_abs(:,3),MU_abs(:,4),'VariableNames', mu_stats_labels);
writetable(mu_table_abs, project_paths('OUT_ANALYSIS','mu_stats_cfnetgross_abs.csv'));
mu_table_rel = table(MU_rel(:,1),MU_rel(:,2),MU_rel(:,3),MU_rel(:,4),'VariableNames', mu_stats_labels);
writetable(mu_table_rel, project_paths('OUT_ANALYSIS','mu_stats_cfnetgross_rel.csv'));

fprintf('Markups relative to winning bid (mean over auctions):\n Inc Mean \t Inc Median \t Ent Mean \t Ent Median\n')
nanmean(MU_rel)
fprintf('Markups relative to winning bid (median over auctions):\n Inc Mean \t Inc Median \t Ent Mean \t Ent Median\n')
nanmedian(MU_rel)
fprintf('Absolute markups (mean over auctions):\n Inc Mean \t Inc Median \t Ent Mean \t Ent Median\n')
nanmean(MU_abs)
fprintf('Absolute markups (median over auctions):\n Inc Mean \t Inc Median \t Ent Mean \t Ent Median\n')
nanmedian(MU_abs)

% Save net cost estimation workspace.
save(project_paths('OUT_ANALYSIS','net_cost_estimation'));