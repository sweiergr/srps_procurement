function [neg_logll, ll] = neg_log_ll(theta,y,X, N, db_win, trunc_point)
% Return the negative of the log-likelihood evaluated at parameter vector
% *theta* taking dependent variable *y* (usually winning bid or margin),
% and regressor matrix *X* for the parameters, vector of number of bidders
% and identity vector for winner
% as additional arguments.

    global N_obs  
    % Evaluate parameters of Weibull bid distribution based on X and theta.
    [parameter_matrix, lambda_vec_I, lambda_vec_E, rho_vec_I, rho_vec_E] = sim_wb_param(X,theta);
    
    % Compute some subparts of the likelihood.
    exp_I = exp(- (y ./ lambda_vec_I).^rho_vec_I);
    exp_E = exp(-  (y ./ lambda_vec_E).^rho_vec_E);
    
    % CDF of Weibull distribution.
    G_I = 1 - exp_I;
    G_E = 1 - exp_E;
    
    % PDF of Weibull distribution.
    g_E = exp_E .* (y ./ lambda_vec_E).^(rho_vec_E-1) .* rho_vec_E./lambda_vec_E;
    g_I = exp_I .* (y ./ lambda_vec_I).^(rho_vec_I-1) .* rho_vec_I./lambda_vec_I;

    % Compute likelihood contributions.
    % For DB winning bids.
    f_bid_db_win = g_I .* (1-G_E).^(N-1);
    % For entrant winning bids.
    f_bid_entrant_win = (N-1) .* g_E .* (1-G_I).*(1-G_E).^(N-2);
    % For all winning bids.
    f_bid_win = db_win .* f_bid_db_win +  (1-db_win) .* f_bid_entrant_win;
    % Take logs and make likelihood negative since we want to minimize.    
    ll = log(f_bid_win);
    neg_logll = -sum(ll);
end