%{
    Estimate cost distributions for incumbent and entrant from gross auctions.
    The estimated bid function parameters are used to compute cost
    distributions for net auction sample.

%}

clear
clc
format('short');

% Define necessary globals.
global data N_obs K
% Set seed in case any simulation is used.
rng(123456);
% Update plots? 1= yes, 0 = no.
update_plots = 1;
% Check cost density for log-concavity.
check_logconcavity = 0;
% SET PAPER SIZE FOR FIGURES TO AVOID LARGE MARGINS.
figuresize(15,12,'cm')
% Prevent graphs from popping up.
set(gcf,'Visible', 'on'); 

%% Read-in data.
fid = fopen(project_paths('OUT_DATA','ga_matlab.csv'),'r');
% Need to manually adjust number of columns of data matrix: Currently 21.
data_raw = textscan(fid, repmat('%s',1,21), 'delimiter',',', 'CollectOutput',true);
data_raw = data_raw{1};
fclose(fid);
% Transform to numerical data plus legend.
data_legend = data_raw(1,:);
data = data_raw(2:end,:);
data = cellfun(@str2double, data);

%% This section can be important since we drop outliers that might distort our bid function estimation.
% Drop extremely large contract: ID 25, line 7, consistent with descriptive statistics.
data(69,:) = [];

%% Extract dependent variable(s) and set of regressors.
% Make all changes to dependent variable and regressors here!
% 2 potential dependent variables: winning bid and winning_margin (bid over
% track costs).
bid_win = data(:,3);
% Set dependent variable: bid (vs. margin).
y = bid_win;
% Set truncation point for bid distribution.
truncation_bid = 0;
% Generate constant.
N_obs = length(y);
cons = ones(N_obs,1);
% Winner's identity and number� of bidders.
db_win = data(:,5);
N = data(:,4);
% Extract number of potential bidders from the last columns of the data matrix.
N_pot_matrix = data(:,end-2:end);

% Truncate number of bidders and winning bids.
N_max = 5 .* cons;
N = min(N,N_max);
bid_min = 0 .* cons;
bid_win = max(bid_win,bid_min);
% Construct trend variable (just for data exploration, currently not used).
trend = (data(:,end) - 1996) ./ 10;

% Construct regressor matrix assuming fully flexible model, i.e. all
% parameters (all lambda and rho regressors) vary by incumbent and
% entrant.)
% IMPORTANT: Use same regressors as for net auctions!
% Column 6: length of track: nkm
% Column 7: total track access charges for full contract
% Column 8: zkm per year for line
% Column 9: frequency of service (constant within lines of a contract)
% Column 10: contract duration in (tens of) years
% Column 15: frequency of service (varying across lines within contract)
% Column 11: zkm per year for full contract (adde dup over all lines within a contract)
% Column 12: dummy for diesel/non-electrified lines.
% Column 13: dummy for used vehicles permitted
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X_orig = [log(data(:,10)), data(:,7), data(:,8), data(:,13), log(data(:,6))./10, log(N)./10];
corr(X_orig)

% Define number of regressors for each (!) parameter.
K = size(X_orig,2);
% Number of total parameters to estimate.
dim_theta = K*4;
% Expand regressor matrix.
X = repmat([repmat(db_win,1,K) .* X_orig, repmat((1-db_win),1,K) .* X_orig],1,2);

%% Estimate parameters of bid function by Maximum Likelihood.
% Set options for various MATLAB optimizers.
fminsearch_options = optimset('Display','iter-detailed', ...
                   'TolFun', 1E-4,'TolX',1E-8,'MaxFunEvals', 17500, 'MaxIter', 15000);
fminunc_options = optimset('Display','iter-detailed','TolFun', 1E-6,'TolX',1E-6, 'MaxFunEvals', 12000, 'MaxIter', 1500);
% Set starting value for theta: very robust to starting values, just make sure to not set them too high for fminunc to work well.       
theta_0 = 0.01 * randn(dim_theta,1);
% Construct anonymous function to pass to minimzers.
min_neg_log_ll = @(theta) neg_log_ll(theta,y,X_orig,N,db_win,truncation_bid);
% Check whether likelihood is defined at starting value.
min_neg_log_ll(theta_0);

% Try with fminsearch to get initial idea on potentail parameter values.
% [theta_opt, neg_log_ll_opt] = fminsearch(min_neg_log_ll, theta_0, fminsearch_options);
% Use fminunc to get more reliable estimate of parameters.
[theta_opt, neg_log_ll_opt] = fminunc(min_neg_log_ll, theta_0,fminunc_options);

% Save vector of coefficient estimates to file.
save(project_paths('OUT_ANALYSIS','ga_est_results'),'theta_opt');
% Compute standard errors and p-values.
[std_errors, t_stats, p_values] = stats_gross(theta_opt, min_neg_log_ll, 10^-8, 2);

% Write estimation results for gross bid functions to file (for quick inspection).
disp('Estimation results for gross auctions:')
results_gross = [theta_opt, std_errors, t_stats, p_values];
col_labels = {'Point Estimates', 'Standard Errors', 't-Statistics', 'p-Values'};
row_labels = {'$\lambda^I_X$', '','','','', '$\lambda^I_N$','$\lambda^E_X$', '', '', '','','$\lambda^E_N$','$\nu^I_X$', '','','','', '$\nu^I_N$','$\nu^E_X$','','','','','$\nu^E_N$'};
matrix2latex(results_gross,project_paths('OUT_TABLES','resultsgross.tex'),'headerRow', col_labels, 'headerColumn', row_labels, 'format', '%6.4f', 'caption', 'Estimation results: bid functions in gross auctions', 'label', 'est_results_gross');
% Save estimation results in raw format to be formatted in a nicer table using Python.
dlmwrite(project_paths('OUT_ANALYSIS','erbf_gross.csv'),results_gross);
fid = fopen(project_paths('OUT_ANALYSIS','erbf_gross_legend.csv'), 'w');
fprintf(fid, '%s,', row_labels{:});
fclose(fid);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Predict and plot bid functions based on theta_opt %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute bid distribution parameters for each line for both incumbent and
% entrant. Combine regressors for incumbent and entrant.
X_aux = X(:,1:K) + X(:,K+1:2*K);
% Compute Weibull bid function parameters.
lambda_I = exp(X_aux * theta_opt(1:K)); 
lambda_E = exp(X_aux * theta_opt(K+1:2*K));
rho_I = exp(X_aux * theta_opt(2*K+1:3*K));
rho_E = exp(X_aux * theta_opt(3*K+1:4*K));

% Define range of bid function and cost plots.
% Bid function plot ranges from 33% to 600% of winning bid.
x_axis_bid = [y ./ 3, y .* 6];
% Cost function plot ranges from 70% to 700% of winning bid.
x_axis_cost = [y ./ 7, y .* 7];

%if update_plots==1
    % Compute parameters for grid.
    % Number of grid points at which to evaluate bid CDF.
    n_bid_grid = 5000;
    lambda_I_grid = repmat(lambda_I,1,n_bid_grid);
    lambda_E_grid = repmat(lambda_E,1,n_bid_grid);
    rho_I_grid = repmat(rho_I,1,n_bid_grid);
    rho_E_grid = repmat(rho_E,1,n_bid_grid);
    % Evaluate Weibull CDF between 0 and 150.
    bid_grid = repmat(linspace(0,150,n_bid_grid),N_obs,1);
    % Compute estimated bid distributions manually.
    % At some point we can switch this to using MATLAB's Weibull CDF function.
    bid_distribution_I = ones(N_obs,n_bid_grid) - exp( - (bid_grid ./ lambda_I_grid) .^ (rho_I_grid) ); 
    bid_distribution_E = ones(N_obs,n_bid_grid) - exp( - (bid_grid ./ lambda_E_grid) .^ (rho_E_grid) ); 
    bid_density_I = exp( - (bid_grid ./ lambda_I_grid)) .^ (rho_I_grid-1) .* (bid_grid ./ lambda_I_grid);
    bid_density_E = exp( - (bid_grid ./ lambda_E_grid)) .^ (rho_E_grid-1) .* (bid_grid ./ lambda_E_grid);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Estimate cost distributions based on estimated bid functions. %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Prepare FOSD test and set its parameters.
% Number of grid points to use for FOSD test.
N_a = 10;
% Test for FOSD on larger range: 0 to 20.
a_grid_full = linspace(0,20,N_a);
% Test for FOSD only on lower tail: 0 to 6. (we may need to update what makes sense here or make this line-specific...
a_grid_lower = linspace(2,6,N_a);
% Collect values of empirical CDFs for costs of incumbent and entrants (at grid points).    
eCDF_cost_full = zeros(N_obs,N_a,2);
eCDF_cost_lower = zeros(N_obs,N_a,2);
% Stack variance of empirical CDF of incumbent and entrant.
cov_FI_FE_full = zeros(N_obs,N_a);
var_a_fosd_full = zeros(N_obs,N_a,4);
cov_FI_FE_lower = zeros(N_obs,N_a);
var_a_fosd_lower = zeros(N_obs,N_a,4);
% Container for test statistic at grid points.
T_grid_full = zeros(N_obs,N_a);
T_grid_lower = zeros(N_obs,N_a);
% Determine critical values manually (they depend on N_a and significance
% level alpha).
% Critical value for singificance level 1%
m_kalpha = 3.29;
% Critical value for singificance level 5%
m_kalpha = 2.8;
% Critical value for singificance level 10%
% m_kalpha = 2.560;
% Container vector for rejection decisions.
% H0: Cost distributions are equal.
% H1: Entrant cost distriubtion dominates incumbent's.
% First column: test full distribution.
% Second column: test only lower range.
reject_equality = zeros(N_obs,2);

%% Initialize containers for counting how many negative costs.
% First row for incumbent, second row for entrant.
neg_costs = zeros(2,N_obs);

%% Prepare cost distribution analysis.
% Set quantile grid at which to compare incumbent and entrant cost function.
quantile_grid = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 0.9, 0.95,0.99];
% Initialize container for cost characteristics for mean and various quantiles.
% Incumbent in first level, entrant in second.
cost_quantiles = zeros(N_obs,size(quantile_grid,2)+1,2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% IMPORTANT "TUNING PARAMETERS" WHEN ESTIMATING COSTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Note that this does not have a direct effect on the net estimation sample.
% Set negative cost threshold: this is the lowest negative cost that we allow.
% With cleaned sample, these thresholds don't matter much.
neg_cost_threshold = -2; 
pos_cost_threshold = 250; 

%% Specification of Kernel density parameters.
% Set number of Kernel density grid points.
N_gridpoints_kd = 500;
% Set number of simulated auctions.
NS_auctions = 5000;
% Set bandwidth for kernel density of cost pdf (not sure this has a big effect).
kd_bandwidth = 0.75;
% Compute bid function parameters for all lines.
[~, lambda_vec_sim_I, lambda_vec_sim_E, rho_vec_sim_I, rho_vec_sim_E] = sim_wb_param(X_orig,theta_opt);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% END OF TUNING PARAMETERS

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
kdens_I = zeros(N_obs,NS_auctions);
kdens_I_grid = zeros(N_obs,NS_auctions);
kCDF_I = zeros(N_obs,NS_auctions);
kCDF_I_grid = zeros(N_obs,NS_auctions);
kdens_E = zeros(N_obs,NS_auctions);
kdens_E_grid = zeros(N_obs,NS_auctions);
kCDF_E = zeros(N_obs,NS_auctions);
kCDF_E_grid = zeros(N_obs,NS_auctions);

%% Initialize containers for markup statistics.
% First two columns indicate statistics  for incumbent (mean and median), last two columns are statistics for entrant. 
MU_abs = zeros(N_obs,4); % absolute value of markup considering full markup distribution.
MU_rel = zeros(N_obs,4); % value of markup relativ to winning bid considering full markup distribution.
MU_abs_clean = zeros(N_obs,4); % absolute value of markup considering distribution of markups but with tails truncated.
MU_rel_clean = zeros(N_obs,4); % value of markup relatve to winning bid considering distribution of markups but with tails truncated.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loop over all gross contracts to compute cost distributions.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for t=1:size(X_orig,1);
    % Extract bid function parameters for line t.
    lambda_I_sim = lambda_vec_sim_I(t,1);
    lambda_E_sim = lambda_vec_sim_E(t,1);
    rho_I_sim = rho_vec_sim_I(t,1);
    rho_E_sim = rho_vec_sim_E(t,1);
    % Simulate bids from estimated bid distributions.
    % Reset random seed for each auction. This is not really necessary.
    rng(123456)
    bid_draw_I = wblrnd(lambda_I_sim,rho_I_sim,[NS_auctions,1]);
    bid_draw_E = wblrnd(lambda_E_sim,rho_E_sim,[NS_auctions,1]);
    
    %% Compute cost signals for each bidder type when evaluated at winning bids.
    % Compute markup term for incumbent.
    % Compute CDF and PDF for both types at winning bid.
    [CDF_I_win, PDF_I_win] = eval_bf(lambda_I_sim,rho_I_sim,bid_win(t,1),truncation_bid);
    [CDF_E_win, PDF_E_win] = eval_bf(lambda_E_sim,rho_E_sim,bid_win(t,1),truncation_bid);
    % Numerator of incumbent's markup term.
    G_MI_win = (1-CDF_E_win).^(N(t)-1);
    % Denominator of entrant's markup term.
    g_MI_win = (N(t)-1) .* (1-CDF_E_win).^(N(t)-2) .* PDF_E_win;
    % Write numerator and denominator into container.
    g_I_win(t,1) = g_MI_win;
    G_I_win(t,1) = G_MI_win;
    % Compute incumbent's markup.
    MU_I_win = G_MI_win ./ g_MI_win;
    % Compute incumbent's winning cost signal.
    RHO_GROSS(t,1) = bid_win(t,1) - MU_I_win;

    % Compute markup term for entrants.
    % Numerator of entrant's markup term.
    G_ME_win = (1-CDF_E_win).^(N(t)-2) .* (1-CDF_I_win);
    % Denominator of entrnat's markup term.
    g_ME_win = (N(t)-2) .* (1-CDF_E_win).^(N(t)-3) .* PDF_E_win  .* (1-CDF_I_win) + PDF_I_win .* (1-CDF_E_win) .^ (N(t)-2);
    % Write numerator and denominator into container.
    G_E_win(t,1) = G_ME_win;
    g_E_win(t,1) = g_ME_win;
    % Compute markup for entrant.
    MU_E_win = G_ME_win ./ g_ME_win;
    % Compute vector of entrants' costs.
    RHO_GROSS(t,2) = bid_win(t,1) - MU_E_win;
            
    % Trim draws from below at 125% of total track access costs. This can be experimented with, but doesn't affect our estimates much.
    bid_draw_I(bid_draw_I < data(t,18)) = 1.25 .* data(t,18);
    bid_draw_E(bid_draw_E < data(t,18)) = 1.25 .* data(t,18);
        
    % Compute markup terms for incumbent.
    % Compute CDF and PDF for both types at incumbent's simulated bid.
    [CDF_I, PDF_I] = eval_bf(lambda_I_sim,rho_I_sim,bid_draw_I,truncation_bid);
    [CDF_E, PDF_E] = eval_bf(lambda_E_sim,rho_E_sim,bid_draw_I,truncation_bid);
    % Numerator of incumbent's markup term.
    G_MI = (1-CDF_E).^(N(t)-1);
    % Denominator of entrant's markup term.
    g_MI = (N(t)-1) .* (1-CDF_E).^(N(t)-2) .* PDF_E;
    % Write numerator and denominator into container.
    g_I(:,t) = g_MI;
    G_I(:,t) = G_MI;
    % Compute incumbent's markup.
    MU_I = G_MI ./ g_MI;
    % Compute vector of incumbent's simulated costs.
    cost_I_grid = bid_draw_I - MU_I;

    %% Check how many cost draws are negative.
    sprintf('Fraction of negative cost draws for incumbent:')
    neg_cost_I = sum(cost_I_grid<0) ./ NS_auctions
    neg_costs(1,t) = neg_cost_I;
    
    % Compute markup terms for entrants.
    % Compute CDF and PDF at entrants' simulated bid.
    [CDF_I, PDF_I] = eval_bf(lambda_I_sim,rho_I_sim,bid_draw_E,truncation_bid);
    [CDF_E, PDF_E] = eval_bf(lambda_E_sim,rho_E_sim,bid_draw_E, truncation_bid);
    % Numerator of entrant's markup term.
    G_ME = (1-CDF_E).^(N(t)-2) .* (1-CDF_I);
    % Denominator of entrnat's markup term.
    g_ME = (N(t)-2) .* (1-CDF_E).^(N(t)-3) .* PDF_E  .* (1-CDF_I) + PDF_I .* (1-CDF_E) .^ (N(t)-2);
    % Write numerator and denominator into container.
    G_E(:,t) = G_ME;
    g_E(:,t) = g_ME;
    % Compute markup for entrant.
    MU_E = G_ME ./ g_ME;
    % Compute vector of entrants' costs.
    cost_E_grid = bid_draw_E - MU_E;
    
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
    graphtitle = sprintf('Comparison of markup distribution for entrant and incumbent for gross contract %d' ,t);
    title(graphtitle);
    legend('Incumbent', 'Entrant')
    filename = sprintf(regexprep(strcat(project_paths('OUT_FIGURES','mu_comp_gross'),num2str(t)),'\','\\\'));
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
    % Negative costs (probably most important for us)
    cost_I_grid(cost_I_grid<neg_cost_threshold) = neg_cost_threshold;
    cost_E_grid(cost_E_grid<neg_cost_threshold) = neg_cost_threshold;
    % Positive costs (probably not relevant for us)
    cost_I_grid(cost_I_grid>pos_cost_threshold) = pos_cost_threshold;
    cost_E_grid(cost_E_grid>pos_cost_threshold) = pos_cost_threshold;
    
    %% Compute bid function plots.
    % Plot incumbent and entrant in one graph.
    subplot(1,1,1)
    CDF_bid_est = plot(bid_grid(t,:), bid_distribution_I(t,:), bid_grid(t,:), bid_distribution_E(t,:));
    graphtitle = sprintf(strcat('Comparison of bid CDF for gross contract ',num2str(t)));
    title(graphtitle);
    legend('Incumbent', 'Entrant')
    axis([x_axis_bid(t,1) x_axis_bid(t,2) 0 1])
    filename = sprintf(regexprep(strcat(project_paths('OUT_FIGURES','gross_bidcdf'),num2str(t)),'\','\\\'));
    saveas(gcf,filename,'pdf');
    
    %% Plot and smooth cost estimates.
    % Safety measure: plot costs only if markup is strictly positive. 
    % This shouldn't be a restrictive if-condition anymore, still keep
    % checking this to be safe.
    if max(G_MI)>0 && max(G_ME)>0
        %% Empirical CDF of costs.
        % Plot empirical CDF of incument's cost in first subplot.
        subplot(2,1,1)
        cdfplot(cost_I_grid);
        title_str = sprintf(strcat('Estimated cost CDF of incumbent for gross contract ',num2str(t)));
        title(title_str)
        axis([x_axis_cost(t,1) x_axis_cost(t,2) 0 1])
        % Plot empirical CDF of entrant's cost in second subplot.
        subplot(2,1,2)
        cdfplot(cost_E_grid);
        title_str = sprintf(strcat('Estimated cost CDF of entrant for gross contract ',num2str(t)));
        title(title_str)
        axis([x_axis_cost(t,1) x_axis_cost(t,2) 0 1])
        filename = sprintf(regexprep(strcat(project_paths('OUT_FIGURES','costcdfgross'),num2str(t)),'\','\\\'));
        saveas(gcf,filename,'pdf');

        %% Kernel densities of costs.
        % Plot kernel density of incumbent's cost in first subplot.
        subplot(2,1,1)
        [kdens_I(t,:), kdens_I_grid(t,:)] = ksdensity(cost_I_grid, bid_grid(t,:),'bandwidth',kd_bandwidth,'NumPoints', N_gridpoints_kd);
        % Compute Kernel CDF for incumbent cost distribution.
        [kCDF_I(t,:),kCDF_I_grid(t,:)] = ksdensity(cost_I_grid,bid_grid(t,:), 'bandwidth',kd_bandwidth,'NumPoints', N_gridpoints_kd ,'Function','cdf');
        % Compute expected cost for incumbent by numerical integration over kernel density.
        cost_quantiles(t,1,1) = trapz(kdens_I_grid(t,:),kdens_I_grid(t,:) .* kdens_I(t,:));
        % Compute cost quantiles of incumbent's cost distribution.
        cq_I = ksdensity(cost_I_grid, quantile_grid, 'bandwidth',kd_bandwidth,'NumPoints', N_gridpoints_kd, 'function','icdf');
        cost_quantiles(t,2:end,1) = cq_I;
        
        %% Do some cosmetics for cost density plot: affects only plots.
        % Truncate incumbent's cost distribution at zero.
        % Scale up so that density integrates to one.
        kdens_I(t,:) = kdens_I(t,:) ./ (1-neg_cost_I);
        % Set negativ density part to zero.
        kdens_I(t,kdens_I_grid(t,:)<0) = 0;
        % Plot kernel density.
        plot(kdens_I_grid(t,:),kdens_I(t,:))
        title_str = sprintf(strcat('Kernel density of incumbent costs for gross contract ',num2str(t)));
        title(title_str)
        axis([x_axis_cost(t,1) x_axis_cost(t,2) 0 0.75])
        % Do the same thing for entrant cost distribution.
        subplot(2,1,2)
        [kdens_E(t,:), kdens_E_grid(t,:)] = ksdensity(cost_E_grid,bid_grid(t,:), 'bandwidth',kd_bandwidth,'NumPoints', N_gridpoints_kd);
        % Compute Kernel CDF for incumbent cost distribution.
        [kCDF_E(t,:),kCDF_E_grid(t,:)] = ksdensity(cost_E_grid,bid_grid(t,:), 'bandwidth',kd_bandwidth,'NumPoints', N_gridpoints_kd ,'Function','cdf');
      
        % Compute expected cost for entrant by numerical integration.
        cost_quantiles(t,1,2) = trapz(kdens_E_grid(t,:),kdens_E_grid(t,:) .* kdens_E(t,:));
        % Compute cost quantiles of entrant's cost distribution.
        cq_E = ksdensity(cost_E_grid, quantile_grid, 'bandwidth',kd_bandwidth,'NumPoints', N_gridpoints_kd, 'function','icdf');
        cost_quantiles(t,2:end,2) = cq_E;
            
        % Truncate cost distribution at zero (same cosmetics as for incumbent).
        kdens_E(t,:) = kdens_E(t,:) ./ (1-neg_cost_E);
        kdens_E(t,kdens_E_grid(t,:)<0) = 0;
        plot(kdens_E_grid(t,:),kdens_E(t,:));
        title_str = sprintf(strcat('Kernel density of entrant costs for gross contract ',num2str(t)));
        title(title_str)
        axis([x_axis_cost(t,1) x_axis_cost(t,2) 0 0.6])
        filename = sprintf(regexprep(strcat(project_paths('OUT_FIGURES','kdensitycostgross'),num2str(t)),'\','\\\'));
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
        % Denominator of entrant's markup term.
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
        title_str = sprintf(strcat('Estimated bid function and cost distribution (incumbent) for gross contract  ',num2str(t)));
        title(title_str,'Interpreter','latex')
        axis([bid_grid(t,1) bid_grid(t,n_bid_grid./10) 0 bid_grid(t,600)])
        yyaxis right;
        plot(bid_grid(t,:), kdens_I(t,:));
        ylabel('Cost density','Interpreter','latex')
        xlabel('Cost Signal (in 10 Mio EUR)','Interpreter','latex')

        % Plot bid function and cost distribution for entrant.
        subplot(2,1,2)
        yyaxis left;
        plot(signal_E_bf(t,:),bid_grid(t,:));
        ylabel('Bid (in 10 Mio EUR)','Interpreter','latex')
        title_str = sprintf(strcat('Estimated bid function and cost distribution (entrant) for gross contract  ',num2str(t)));
        title(title_str,'Interpreter','latex')
        axis([bid_grid(t,1) bid_grid(t,n_bid_grid./10) 0 bid_grid(t,600)])
        yyaxis right;
        plot(bid_grid(t,:), kdens_E(t,:));
        ylabel('Cost density','Interpreter','latex')
        xlabel('Cost Signal (in 10 Mio EUR)','Interpreter','latex')
        filename = sprintf(regexprep(strcat(project_paths('OUT_FIGURES','bfkd_gross'),num2str(t)),'\','\\\'));
        saveas(gcf,filename,'pdf');
        
        % Verify that cost distribution is log-concave.
        if check_logconcavity==1
            subplot(2,1,1)
            [kdens, kdens_grid] = ksdensity(cost_I_grid,'bandwidth',0.5,'NumPoints', N_gridpoints_kd);
            log_kdens = log(kdens);
            plot(kdens_grid,log_kdens)
            title_str = sprintf(strcat('Density of incumbent log-costs for gross contract ',num2str(t)));
            title(title_str,'Interpreter','latex')
            axis([0 15 0 0.6])
            subplot(2,1,2)
            [kdens, kdens_grid] = ksdensity(cost_E_grid,'bandwidth',0.5,'NumPoints', N_gridpoints_kd);
            log_kdens = log(kdens);
            plot(kdens_grid,log_kdens);
            title_str = sprintf(strcat('Density of entrant log-costs for gross contract ',num2str(t)));
            title(title_str,'Interpreter','latex')
            axis([0 15 0 0.6])
            filename = sprintf(regexprep(strcat(project_paths('OUT_FIGURES','logdensitycostgross'),num2str(t)),'\','\\\'));
            saveas(gcf,filename,'pdf');
        end

        %% Compute Davidson-Duclos FOSD test.
        % Test for FOSD on full range: 0 to 90%-percentile of bid function.
        c_full_max = max(cost_quantiles(t,8,1),cost_quantiles(t,8,2));
        c_lower_max = max(cost_quantiles(t,3,1),cost_quantiles(t,3,2));
        a_grid_full = linspace(0,8*bid_win(t),N_a);
        % Test for FOSD only on lower tail: 0 to 6. (we may need to update what makes sense here or make this line-specific...)
        a_grid_lower = linspace(0,2*bid_win(t),N_a);
        
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
end

%% Analyze mean markup components for each contract. This helps debugging where markup term becomes unreliable.
g_I_mean = mean(g_I);
G_I_mean = mean(G_I);
g_E_mean = mean(g_E);
G_E_mean = mean(G_E);

%% Post-estimation analysis of cost estimates.
% Median costs for each contract and bidder type.
% Incumbent in first column, entrant in second column.
clf('reset')
% SET PAPER SIZE FOR FIGURES TO AVOID LARGE MARGINS.
figuresize(15,12,'cm')
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
saveas(gcf,project_paths('OUT_FIGURES','rca_gross'),'pdf');
% Save median and mean cost data to join with net sample.
median_costs_gross = median_costs;
mean_costs_gross = mean_costs;
rel_cost_adv_med_gross = rel_cost_adv_med;
rel_cost_adv_gross = rel_cost_adv;
save(project_paths('OUT_ANALYSIS','rca_data_gross.mat'),'median_costs_gross', 'mean_costs_gross', 'rel_cost_adv_med_gross', 'rel_cost_adv_gross');

% When reporting these results we might have to adjust/drop outliers.
% Truncate negative costs at negative cost threshold.
outlier_indicator_below = max((mean_costs(:,1) < neg_cost_threshold),(mean_costs(:,2) < neg_cost_threshold));
% Truncate positive costs at upper threshold.
outlier_indicator_above = max((mean_costs(:,1) > pos_cost_threshold),(mean_costs(:,2) > pos_cost_threshold));
% Construct compound outlier indicator.
outlier_indicator = max(outlier_indicator_below,outlier_indicator_above);
% Drop outliers from computation of mean costs for bidder types. With cleaned sample this doesn't matter much.
cost_quantiles_clean = (cost_quantiles(outlier_indicator==0,:,:));
avg_cost_quantiles = mean(cost_quantiles_clean);
% Compute difference in quantiles between incumbent and entrant.
avg_cq_diff = avg_cost_quantiles(:,:,1) - avg_cost_quantiles(:,:,2);
% Incumbent probably has advantage on median and average costs.
med_cost_advantage = -mean(avg_cq_diff(:,7));
mean_cost_advantage = -mean(avg_cq_diff(:,1));
med_cost_avg = mean(cost_quantiles_clean(:,7,:));
mean_cost_avg = mean(cost_quantiles_clean(:,1,:));
% Save cost data for gross auction workspace.
cq_clean_gross = cost_quantiles_clean;
save(project_paths('OUT_ANALYSIS','cq_clean_gross.mat'));

%% Compare median and mean costs to winning bids: Potentially an indicator for how profitable public procurement is. Compute shading factor, i.e. median costs vs. winning bid.
shading_median = [cost_quantiles(:,8,1), cost_quantiles(:,8,2)];
shading_median_I = (bid_win -  cost_quantiles(:,8,1)) ./ cost_quantiles(:,8,1) .* db_win;
shading_median_E = (bid_win -  cost_quantiles(:,8,2)) ./ cost_quantiles(:,8,2) .* (1-db_win);

shading_median_I = shading_median_I(shading_median_I~=0);

% Truncate extreme realizations to mitigate effect of outliers.
shading_median_I(shading_median_I<-5) = -5;
shading_median_I(shading_median_I>5) = 5;
shading_median_E(shading_median_E<-2) = -2;
shading_median_E(shading_median_E>2) = 2;

shading_median_E = shading_median_E(shading_median_E~=0);
subplot(2,1,1)
hist(shading_median_I)
title('Histogram of comparison of winning bid and median costs for incumbent','Interpreter','latex')
subplot(2,1,2)
hist(shading_median_E)
title('Histogram of comparison of winning bid and median costs for incumbent','Interpreter','latex')
filename = project_paths('OUT_FIGURES','shadingmediangross');
saveas(gcf,filename,'pdf');

% Compute shading factor, i.e. mean costs vs. winning bid.
shading_mean = [cost_quantiles(:,1,1), cost_quantiles(:,1,2)];
shading_mean_I = (bid_win -  cost_quantiles(:,1,1)) ./ cost_quantiles(:,1,1) .* db_win;
shading_mean_E = (bid_win -  cost_quantiles(:,1,2)) ./ cost_quantiles(:,1,2) .* (1-db_win);

shading_mean_I = shading_mean_I(shading_mean_I~=0);
% Truncate extreme realizations to reduce effects of outliers.
shading_mean_I(shading_mean_I<-5) = -5;
shading_mean_I(shading_mean_I>5) = 5;
shading_mean_E(shading_mean_E<-2) = -2;
shading_mean_E(shading_mean_E>2) = 2;

shading_mean_E = shading_mean_E(shading_mean_E~=0);
subplot(2,1,1)
hist(shading_mean_I)
title('Histogram of comparison of winning bid and mean costs for incumbent','Interpreter','latex')
subplot(2,1,2)
hist(shading_mean_E)
title('Histogram of comparison of winning bid and mean costs for incumbent','Interpreter','latex')
filename = project_paths('OUT_FIGURES','shadingmeangross');
saveas(gcf,filename,'pdf');

% Sum of lines for which we can reject equality of distributions in favor
% of entrant stochastically dominating incumbent.
sum_reject_Fcost_equality = sum(reject_equality);
share_reject_Fcost_equality = sum_reject_Fcost_equality ./ size(reject_equality,1);

str_logtitle = sprintf(strcat('Summary statistics for FOSD test of equality of cost distribution in gross auctions\nEstimated on:\n', date,'\n\n'));
fid = fopen(project_paths('OUT_ANALYSIS','fosd_test_gross.txt'),'wt');
fprintf(fid,str_logtitle);
fprintf(fid,':SUMMARY STATISTICS WHEN TESTING FOR FOSD OF COST DISTRIBUTIONS IN GROSS AUCTION SAMPLE:\n')
fprintf(fid,'Reject equality of entire distribution in favor of incumbent having better cost distribution on %d gross auction lines.\n ', sum_reject_Fcost_equality(1));
fprintf(fid,'Reject equality of lower part of distribution in favor of incumbent having better cost distribution on %d gross auction lines.\n ', sum_reject_Fcost_equality(2));
fprintf(fid,'\nEXPRESSED IN SAHRE OF TOTAL NUMBER OF LINES IN GROSS SAMPLE:\n')
fprintf(fid,'Reject equality of entire distribution in favor of incumbent having better cost distribution on %6.4f percent of gross auction lines.\n ', 100 .* share_reject_Fcost_equality(1));
fprintf(fid,'Reject equality of lower part of distribution in favor of incumbent having better cost distribution on %6.4f percent of gross auction lines.\n ', 100 .* share_reject_Fcost_equality(2));
fclose(fid);

% Print mean of negative costs for incumbent and entrant.
max(neg_costs')
mean(neg_costs,2)
% Statistics of winner's cost signal distribution.
max(RHO_GROSS)
mean(RHO_GROSS)
min(RHO_GROSS)
% Plot empirical CDF of incument's and entrants' cost conditional upon winning
% (pooling over all contract characteristics and number of bidders).
subplot(2,1,1)
RHO_GROSS_I = RHO_GROSS(:,1);
costgross_win_I_plot = RHO_GROSS_I(db_win==1);
cdfplot(costgross_win_I_plot);
title('Gross: Cost signal distribution of winning incumbent','Interpreter','latex')
axis([-30 30 0 1])
subplot(2,1,2)
RHO_GROSS_E = RHO_GROSS(:,2);
costgross_win_E_plot = RHO_GROSS_E(db_win==0);
cdfplot(costgross_win_E_plot);
title('Gross: Cost signal distribution of winning entrant','Interpreter','latex')
axis([-30 30 0 1])
filename = project_paths('OUT_FIGURES','costwingross');
saveas(gcf,filename,'pdf');

% Plot histogram.
subplot(2,1,1)
hist(MU_abs_clean(:,[1,3]));
legend('Incumbent','Entrant');
title('Histogram of mean (absolute) markups');
subplot(2,1,2)
hist(MU_rel_clean(:,[1,3]));
legend('Incumbent','Entrant');
title('Histogram of mean markups (relative to winning bid)');
saveas(gcf,project_paths('OUT_FIGURES','hist_mean_mu_gross'),'pdf');

subplot(2,1,1)
hist(MU_abs_clean(:,[2,4]));
legend('Incumbent','Entrant');
title('Histogram of median (absolute) markups');
subplot(2,1,2)
hist(MU_rel_clean(:,[2,4]));
legend('Incumbent','Entrant');
title('Histogram of median markups (relative to winning bid)');
saveas(gcf,project_paths('OUT_FIGURES','hist_median_mu_gross'),'pdf');

% Write markup statistic to table.
% Two files for absolute and relative markup statistics.
mu_stats_labels = {'I_mean','I_median','E_mean','E_median'};
mu_table_abs = table(MU_abs(:,1),MU_abs(:,2),MU_abs(:,3),MU_abs(:,4),'VariableNames', mu_stats_labels);
writetable(mu_table_abs, project_paths('OUT_ANALYSIS','mu_stats_gross_abs.csv'));
mu_table_rel = table(MU_rel(:,1),MU_rel(:,2),MU_rel(:,3),MU_rel(:,4),'VariableNames', mu_stats_labels);
writetable(mu_table_rel, project_paths('OUT_ANALYSIS','mu_stats_gross_rel.csv'));

fprintf('Markups relative to winning bid (mean over auctions):\n Inc Mean \t Inc Median \t Ent Mean \t Ent Median\n')
nanmean(MU_rel)
fprintf('Markups relative to winning bid (median over auctions):\n Inc Mean \t Inc Median \t Ent Mean \t Ent Median\n')
nanmedian(MU_rel)
fprintf('Absolute markups (mean over auctions):\n Inc Mean \t Inc Median \t Ent Mean \t Ent Median\n')
nanmean(MU_abs)
fprintf('Absolute markups (median over auctions):\n Inc Mean \t Inc Median \t Ent Mean \t Ent Median\n')
nanmedian(MU_abs_clean)

% Save gross auction estimation workspace for use in counterfactuals.
save(project_paths('OUT_ANALYSIS','grossauction_workspace'));