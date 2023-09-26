%{
   Compute and plot net bid functions as function of own signal rho.

%}

clear
clc
format('short');
% Define necessary globals.
global N_obs K
% Set seed in case any simulation is used.
rng(123456);
% Indicate what to do with existing plots. Set to 1, if you want to generate the bid function graphs.
update_plots = 1;

%% Load net auction workspace and gross auction parameters.
load(project_paths('OUT_ANALYSIS','postestimation_workspace_net'));
% Plot estimated bid distributions for each contract.
clf('reset')
figuresize(15,12,'cm')

% Set up bid grid.
% Set number of Kernel density grid points.
n_bid_grid = 60;
bid_grid = zeros(N_obs, n_bid_grid);
% Compute bid_grid differently for each line.
for t=1:N_obs
    bid_grid(t,:) = linspace(1.5 .* bid_win(t) , 8 .* bid_win(t) ,n_bid_grid);
end
% Set bandwidth for kernel density of cost pdf (usually, this does not have a very big  effect).
kd_bandwidth = 0.5;
% Compute bid function parameters for all lines.
[~, lambda_vec_sim_I, lambda_vec_sim_E, rho_vec_sim_I, rho_vec_sim_E] = sim_wb_param(X_orig,theta_opt);

% Construct container for markup components (useful for checking where markups become implausible).
% G_I = zeros(NS_auctions, N_obs);
% g_I = zeros(NS_auctions, N_obs);
% G_E = zeros(NS_auctions, N_obs);
% g_E = zeros(NS_auctions, N_obs);

% Containers for winning bid signals.
g_I_win = zeros(N_obs,1);
G_I_win = zeros(N_obs,1);
g_E_win = zeros(N_obs,1);
G_E_win = zeros(N_obs,1);

% Containers for winning bid signals.
g_I_bf = zeros(N_obs,n_bid_grid);
G_I_bf = zeros(N_obs,n_bid_grid);
g_E_bf = zeros(N_obs,n_bid_grid);
G_E_bf = zeros(N_obs,n_bid_grid);
% Container for RHO-signal.
signal_I_bf = zeros(N_obs,n_bid_grid);
signal_E_bf = zeros(N_obs,n_bid_grid);
% Container for own signal: rho.
signal_I_rho_bf = zeros(N_obs,n_bid_grid);
signal_E_rho_bf = zeros(N_obs,n_bid_grid);

% Compute revenue mean, variane and alpha parameters ot put into X_IE
% routine.
% These should be mean and variance of the parent normal distribution.
R_nbf = mean_rev_aux;
sigma_r_bnf = sigma_rev_aux.^2;
alpha_bnf = alpha_N(:,1:2);
% Container for E-term in RHO-signal.
E_term = zeros(N_obs,n_bid_grid,2);
E_term_I = zeros(N_obs,n_bid_grid);
E_term_E = zeros(N_obs,n_bid_grid);
% Initialize container.
X_IE_bnf = zeros(N_obs,4,n_bid_grid);
X_IE_bnf_test = zeros(N_obs,4,n_bid_grid);
% Set options for solving for X_IE.
% Restrict solution to be nonnegative.
% Update this to same number as in estimation?
lb = [-1;-1;-1;-1];
% Set options for solver.
options=optimoptions('lsqnonlin','Display','off');

for t=1:N_obs
    % Extract relevant bid function parameters.
    lambda_I_sim = lambda_vec_sim_I(t,1); 
    rho_I_sim = rho_vec_sim_I(t,1);
    lambda_E_sim = lambda_vec_sim_E(t,1);
    rho_E_sim = rho_vec_sim_E(t,1);
    
    % Compute markups for each bid.
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
    % CHECK HOW TO CORRECT COMPUTED SIGNALS.
    signal_I_bf_aux(G_I_bf(t,:)==0) = bid_grid(t,G_I_bf(t,:)==0);
    signal_I_bf(t,:) = signal_I_bf_aux;
    % Compute markup term for entrants.
    % Numerator of entrant's markup term.
    G_ME_bf = (1-CDF_E_win).^(N(t)-2) .* (1-CDF_I_win);
    % Denominator of entrnat's markup term.
    if N(t)==2
        g_ME_bf = PDF_I_win .* (1-CDF_E_win).^(N(t)-2);
    else
        g_ME_bf = (N(t)-2) .* (1-CDF_E_win).^(N(t)-3) .* PDF_E_win  .* (1-CDF_I_win) + PDF_I_win .* (1-CDF_E_win) .^ (N(t)-2);
    end
    % Write numerator and denominator into container.
    G_E_bf(t,:) = G_ME_bf;
    g_E_bf(t,:) = g_ME_bf;
    % Compute markup for entrant.
    MU_E_bf = G_ME_bf ./ g_ME_bf;
    % Compute vector of entrants' costs.
    signal_E_bf_aux = bid_grid(t,:) - MU_E_bf;
    signal_E_bf_aux(G_E_bf(t,:)==0) = bid_grid(t,G_E_bf(t,:)==0);
    signal_E_bf(t,:) = signal_E_bf_aux;
    % Loop over points in bid grid.        
        for b=1:size(bid_grid,2)
            % Adjust signals for X_IE to get rho.
            % Compute X_IE for each grid point.
            X_IE_start = [0.4.*R_nbf(t);0.2 .* R_nbf(t);0.2.*R_nbf(t);0.1 .* R_nbf(t)];
            RHO_signal = [signal_I_bf(t,b); signal_E_bf(t,b)];
            % Anonymous function to input data and auxiliary parameters.
            solve_E_FP_a = @(X_IE) solve_E_FP(X_IE, alpha_bnf(t,2), alpha_bnf(t,1), R_nbf(t), sigma_r_bnf(t), RHO_signal, kdens_container(:,t,:,:), N(t),db_prob(t));
            % Compute fixed point of system describing beliefs about rivals'
            % revenue signals: use lsqnonlin instead of fsolve to impose
            % nonnegativity constraints/
            X_IE_bnf(t,:,b) = lsqnonlin(solve_E_FP_a,X_IE_start,lb,[],options);
            X_IE_bnf_test(t,:,b) = solve_E_FP_a(X_IE_bnf(t,:,b));
            % For incumbent.
            E_term(t,b,1) = (N(t)-2) .* alpha_bnf(t,2) .* X_IE_bnf(t,4,b) + alpha_bnf(t,2) .* X_IE_bnf(t,2,b);
            % For entrant.
            if N(t)==2 % distinguish special case for E-term when only 2 bidders.
                E_term(t,b,2) = alpha_bnf(t,1) .* X_IE_bnf(t,1,b);
            elseif N(t)>2
                E_term(t,b,2) = db_prob(t) .*  (alpha_bnf(t,1) .* X_IE_bnf(t,1,b) + (N(t)-2) .* alpha_bnf(t,2) .* X_IE_bnf(t,4,b)) + ...
                (1-db_prob(t)) .* (alpha_bnf(t,1) .* X_IE_bnf(t,3,b) + alpha_bnf(t,2) .* X_IE_bnf(t,2,b) + (N(t)-3) .* alpha_bnf(t,2) .* X_IE_bnf(t,4,b));  
            end
        end
        % Copy E-terms to separate vectors for entrant and incumbent for
        % investigation.
        E_term_I(t,:) = E_term(t,:,1);
        E_term_E(t,:) = E_term(t,:,2);
        % Adjust RHO for X_IE values.
        signal_I_rho_bf(t,:) = signal_I_bf(t,:) + E_term(t,:,1);
        signal_E_rho_bf(t,:) = signal_E_bf(t,:) + E_term(t,:,2);
        
        % Plot bid function and cost distribution for incumbent.
        subplot(2,1,1)
        plot(signal_I_rho_bf(t,:),bid_grid(t,:));
        ylabel('Bid (in 10 Mio EUR)', 'Interpreter', 'latex')
        title_str = sprintf(strcat('Estimated net bid function (incumbent) for net contract  ',num2str(t)));
        title(title_str, 'Interpreter', 'latex')
        axis([bid_grid(t,1)./2 bid_grid(t,n_bid_grid./2) 0 bid_grid(t,n_bid_grid./2)])
        xlabel('Net Cost Signal ($\rho$) in 10 Mio EUR', 'Interpreter', 'latex')
        
        % Plot bid function and cost distribution for entrant.
        subplot(2,1,2)
        plot(signal_E_rho_bf(t,:),bid_grid(t,:));
        ylabel('Bid (in 10 Mio EUR)', 'Interpreter', 'latex')
        title_str = sprintf(strcat('Estimated net bid function (entrant) for net contract  ',num2str(t)));
        title(title_str, 'Interpreter', 'latex')
        axis([bid_grid(t,1)./2 bid_grid(t,n_bid_grid./2) 0 bid_grid(t,n_bid_grid./2)])
        xlabel('Net Cost Signal ($\rho$) in 10 Mio EUR','Interpreter','latex')
        filename = sprintf(regexprep(strcat(project_paths('OUT_FIGURES','bf_net_rho'),num2str(t)),'\','\\\'));
        saveas(gcf,filename,'pdf');
       
        % Plot bid function and cost distribution for incumbent.
        subplot(1,1,1)
        plot(signal_I_rho_bf(t,:),bid_grid(t,:),signal_E_rho_bf(t,:),bid_grid(t,:));
        ylabel('Bid (in 10 Mio EUR)', 'Interpreter', 'latex')
        legend('Incumbent', 'Entrant')
        axis([bid_grid(t,1)./2 bid_grid(t,n_bid_grid./2) 0 bid_grid(t,n_bid_grid./2)])
        title_str = sprintf(strcat('Comparison of net bid functions for net contract  ',num2str(t)));
        title(title_str, 'Interpreter', 'latex')
        xlabel('Net Cost Signal ($\rho$) in 10 Mio EUR', 'Interpreter', 'latex')
         filename = sprintf(regexprep(strcat(project_paths('OUT_FIGURES','bf_net_rho_comb'),num2str(t)),'\','\\\'));
       saveas(gcf,filename,'pdf');       
end