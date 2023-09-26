%{
    Compute hypothetical bid functions and expected revenues of gross
    tracks.

%}

clear
clc
format('short');
% Define necessary globals.
global N_obs K

% Indicate what to do.
update_plots = 1;
% Prevent graphs from popping up.
set(gcf,'Visible', 'on'); 
% SET PAPER SIZE FOR FIGURES TO AVOID LARGE MARGINS.
figuresize(15,12,'cm')

% Set seed in case any simulation is used.
rng(123456);

%% Load gross auction workspace and net auction parameters.
load(project_paths('OUT_ANALYSIS','grossauction_workspace'));
%% Compute hypothetical bid functions: gross -> net
results_net = dlmread(project_paths('OUT_ANALYSIS','erbf_net.csv'));
theta_est_net = results_net(:,1);

% Define number of observations.
T = length(bid_win);
% Compute matrix of bid function parameters.
% Compute hypothetical bid distribution parameters for each line for both incumbent and
% entrant. Combine regressors for incumbent and entrant.
X_aux = X(:,1:K) + X(:,K+1:2*K);
lambda_I_gn = exp(X_aux * theta_est_net(1:K)); 
lambda_E_gn = exp(X_aux * theta_est_net(K+1:2*K));
rho_I_gn = exp(X_aux * theta_est_net(2*K+1:3*K));
rho_E_gn = exp(X_aux * theta_est_net(3*K+1:4*K));

% Plot bid hypothetical bid functions for all lines.
if update_plots==1
    figuresize(15,12,'cm')
    plot_bid_functions(lambda_I_gn, lambda_E_gn, rho_I_gn , rho_E_gn , 'Hypothetical bid distributions for gross contracts as net', ...
                        'ga_hyp_bf',[0,20]);
end                

%% Compute expected ticket revenues for both gross and net auction sample.
% Load net 2-step parameters.
results_net2 = dlmread(project_paths('OUT_ANALYSIS','er_net2.csv'));
theta_net2 = results_net2(:,1);

X_revenue = [X_orig(:,4), data(:,7)];
X_reg = [ones(T,1), X_revenue];
R = X_reg * theta_net2(5:end,1);
