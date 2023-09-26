%{
    Compute average markups for different bidder types for our gross
    auction sample.

%}

clear
clc
format('short');
% Define necessary globals.
global data N_obs K
% Set seed in case any simulation is used.
rng(123456);
% Load gross auction estimation workspace.
load(project_paths('OUT_ANALYSIS','grossauction_workspace'));
% Construct containers for average markups for each auction and bidder
% type.
MU_average = zeros(N_obs,3);

% Loop over all gross auctions.
for t=1:N_obs
    % Throughout set truncation bid to zero.
    trunaction_bid = 0;
    % Extract bid function parameters for line t.
    lambda_I_sim = lambda_vec_sim_I(t,1);
    lambda_E_sim = lambda_vec_sim_E(t,1);
    rho_I_sim = rho_vec_sim_I(t,1);
    rho_E_sim = rho_vec_sim_E(t,1);

    % Construct bid grid that covers full range of bid distribution from 0.05
    % to 99.5 percent.
    % How many grid points to discretize to.
    n_grid = 500;
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
    test_pdf_E = trapz(bid_grid_E,PDF_E);
    % Integrate over bid density.
    MU_average(t,2) = trapz(bid_grid_E,PDF_E.*MU_E);
end 

% Ratio of average markups for each line.
MU_average(:,3) = MU_average(:,2) ./ MU_average(:,1);

fprintf('Median markup of incumbent: %6.4f. \n', median(MU_average(:,1)));
fprintf('Median markup of entrant: %6.4f. \n', median(MU_average(:,2)));
fprintf('Median ratio of entrant over incumbent markup: %6.4f. \n', median(MU_average(:,3)));

% Export data to file.
% Export data to csv file for reformatting in Python.
VarNames_mu_gross = {'MU_I','MU_E','ratio'};
mu_data_gross = table(MU_average(:,1), MU_average(:,2), MU_average(:,3), ...
			  'VariableNames', VarNames_mu_gross);
% Export efficiency data to to csv-file.
writetable(mu_data_gross, project_paths('OUT_ANALYSIS','average_mu_gross.csv'));
