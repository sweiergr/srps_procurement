function [neg_ll, ll,X_IE,n_ll_problematic] = log_ll_net_2step(theta, X_rev, RHO, db_win, kdens_container, N, db_not_old, db_prob)
%{
       This function sets up the log likelihood of the second step of the
       net auction estimation.
       The parameters of the likelihood are stacked as follows in the
       vector * theta*:
            alpha: asymmetry parameter
            sigma: standard deviation parameter of the revenue signal distribution
            R: mean parameter of the revenue signal distribution. This
            should eventually be a function of track characteristics, such
            as frequency of service.
       The other (auxiliary inputs of the likelihood are:
            RHO: the vector of net cost signals from the first step. This
            is the net cost signal that induces each bidder type to bid the
            winning bid.
            cost_grid: an estimate for the cost functions for each bidder
            type and each track.
            db_prob: Vector of probabilities for DB being pivotal bidder
            evaluated at observed winning bid (used for computation of
            conditional revenue expectation of other bidders).
        Function argument db_not_old is only used for debugging and testing purposes. Not relevant for final version of the code.

    %}

global theta_debug

    % Number of observations/auctions.
    T = length(db_win);
    % Number of cost grid points.
    no_cost_gp = size(kdens_container,1);
    % Create several containers.
    likelihood = zeros(T,1);
    % Cost grid and density.
    c_grid = zeros(no_cost_gp,T);
    c_dens = c_grid;
    c_dens_cond = c_dens;
    % Revenue signal as function of RHO, c and X_IE.
    f_r_arg = zeros(no_cost_gp,T);
    % Revenue signal density evaluated at argument grid.
    f_r_aux = zeros(no_cost_gp,T);
    % Revenue signal density corrected for truncation at zero.
    f_r = zeros(no_cost_gp,T);
    % Joint density of revenue and cost signal.
    f_cr = zeros(no_cost_gp,T);
    % Initialize container for expectation terms.
    X_IE = zeros(T,4);
    % Only for debugging and testing.
    X_IE_test = zeros(T,4);

    E_term_sum = zeros(T,1);
    % Containers for additional fsolve diagnostics.
    options_fsolve = optimset('Display','off');
       
    % Extract information from kdens-container.
    % For incumbent.
    cost_density_I = kdens_container(:,:,1,1);
    cost_grid_I = kdens_container(:,:,1,2);
    % For entrant.
    cost_density_E = kdens_container(:,:,2,1);
    cost_grid_E = kdens_container(:,:,2,2);
    
    %% Construct parameter vectors and regressor matrices.
    % Non-parametric version of alpha
    alpha_1 = theta(1);
    alpha_2 = theta(2);
    alpha_3 = theta(3);
    alpha_4 = theta(4);

    % Extract parameters for sigma and r.
    sigma_0 = theta(end-(size(X_rev,2)+1));
    r_vec = theta(end-size(X_rev,2):end);
    % Construct winning alpha for distinct N-groups.
    % Non-parametric version with logistic transformation.
    alpha_I = ...
        exp(alpha_1) ./ (1.0 + exp(alpha_1)) .* (N==2) + ...
        exp(alpha_2) ./ (1.0 + exp(alpha_2)) .* (N==3) + ...
        exp(alpha_3) ./ (1.0 + exp(alpha_3)) .* (N==4) + ...
        exp(alpha_4) ./ (1.0 + exp(alpha_4)) .* (N==5);
    % Compute alpha_E as residual.
    alpha_E = (ones(T,1) - alpha_I) ./ (N-1);
    % Construct winning alpha vector.
    alpha = db_win .* alpha_I + (1-db_win) .* alpha_E;

    % Variance of revenue signal distribution.
    % sigma_r has to be positive; therefore, use exponential transformation.
    sigma_r = exp(sigma_0) .* ones(T,1);
    
    % Mean of revenue signal as function of revenue regressors.
    X_reg = X_rev;
    % Keep in mind that R is the mean of the parent normal, not the
    % truncated normal, therefore, restricting to positive values is not
    % necessary.
    R = ([ones(T,1) X_reg] * r_vec);
    % Randomly perturb Mean revenue.
    % Distinguish between own rho and RHO matrix (fix this).
    RHO_WIN = db_win .* RHO(:,1) + (1-db_win) .* RHO(:,2);
    X_start_eq = 0.5 .* ([mean(cost_grid_I)' mean(cost_grid_E)'] -  RHO - [alpha_I alpha_E] .* repmat(R,1,2)) ./ repmat((N-1),1,2);
    X_start_plus = 0.1 .* X_start_eq;
    X_start = [X_start_eq X_start_plus];
    %% Compute loglikelihood.
    % Loop over all net auctions.
    for t=1:T
        t;
        % Extract information for computing E-terms.
        C = kdens_container(:,t,:,:);
        % Starting values for conditional expectations set to unconditional
        % mean. 
        X_IE_start = X_start(t,:)';
        % Anonymous function to input data and auxiliary parameters.
        solve_E_FP_a = @(X_IE) solve_E_FP(X_IE, alpha_E(t), alpha_I(t), R(t), sigma_r(t), RHO(t,:)', C , N(t),db_prob(t));
        % Compute fixed point of system describing beliefs about rivals'
        % revenue signals: use lsqnonlin instead of fsolve to impose
        % nonnegativity constraints.
        % Restrict solution to be nonnegative might be too strong, let's just avoid overly negative values..
        lb = [-100;-100;-100;-100];
        %lb = [0;0;0;0];
        
        % Set options for solver.
        options=optimoptions('lsqnonlin','Display','off');
        X_IE_t= lsqnonlin(solve_E_FP_a,X_IE_start,lb,[],options);
        % Only for debugging/testing.
        X_IE(t,:) = X_IE_t;
        % Recall new order of terms in fixed point search:
        % 1. X_I_eq
        % 2. X_E_eq
        % 3. X_I_plus
        % 4. X_E_plus

        % Construct sums for decomposing compound signal RHO.
        % E-revenue term for incumbent.
        E_term_sum_I =  alpha_E(t) .* ((N(t)-2) .* X_IE(t,4) + X_IE(t,2));
        % E-revenue term for entrants.
        if N(t)==2 % special case if there are only 2 bidders.
            E_term_sum_E = alpha_I(t) .* X_IE(t,1);
        elseif N(t)>2
            E_term_sum_E = db_prob(t,1) .*  ((N(t)-2) .* alpha_E(t) .* X_IE(t,4) + alpha_I(t) .* X_IE(t,1)) + ...
                           (1-db_prob(t,1)) .*  ((N(t)-3) .* alpha_E(t) .* X_IE(t,4) + alpha_E(t) .* X_IE(t,2) + alpha_I(t) .* X_IE(t,3));
        end
        % Combine terms for incumbent and entrant winning.
        E_term_sum(t,1) = db_win(t) .* E_term_sum_I + ... case where DB wins
                          (1-db_win(t)) .* E_term_sum_E; % case where entrant wins
        
        % Replicate to size of cost grid.
        E_term_sum_vec = repmat(E_term_sum(t,1),no_cost_gp,1);
        
        % Cost densities and grids for winner
        c_grid(:,t) = db_win(t) .* cost_grid_I(:,t) + (1-db_win(t)) .* cost_grid_E(:,t);
        c_dens(:,t) = db_win(t) .* cost_density_I(:,t) + (1-db_win(t)) .* cost_density_E(:,t);
        
        % Debugging and testing: Check that RHO is not larger than maximum cost.
        winner_max_cost = db_win(t,1) .* max(cost_grid_I(:,t)) + (1-db_win(t,1)) .* max(cost_grid_E(:,t));
        if RHO_WIN(t,1) > winner_max_cost
            RHO_WIN(t,1) = 0.95 .* (winner_max_cost-E_term_sum(t,1));
        end
        % Net cost signal for winner.
        rho_vector = repmat(RHO_WIN(t,1),no_cost_gp,1);
        % Compute density of revenue part of signal.        
        % Own revenue signal as function of c, RHO and E-terms.
        f_r_arg(:,t) = (c_grid(:,t) - rho_vector - E_term_sum_vec) ./ alpha(t);
        % Evaluate density of revenue signal based on truncated normal.
        f_r_aux(:,t) = normpdf((f_r_arg(:,t)-R(t))./sigma_r(t)) ./ ( alpha(t) .* sigma_r(t) ) ...
                           ./ normcdf(R(t)./sigma_r(t));
        % Combine zero and non-zero parts of density of revenue signal.          
        f_r(:,t) = f_r_aux(:,t) .* (f_r_arg(:,t) >=0) + 0 .* (f_r_arg(:,t) < 0);
        cd_cond_aux = c_dens(:,t);
        cd_cond_aux(f_r_arg(:,t)<0) = 0;
        mass_ex_cost = sum(c_dens(f_r_arg(:,t)<0)) ./ sum(c_dens(:,t));
        cd_cond_aux(f_r_arg(:,t)>=0) = c_dens(f_r_arg(:,t)>0) ./ (1-mass_ex_cost);
        c_dens_cond(:,t) = cd_cond_aux;
        % Joint density of revenue and cost density.
        % With unconditional cost distribution.
        f_cr(:,t) = f_r(:,t) .* c_dens(:,t);
        % Numerically integrate over cost grid using trapzoid method.
        likelihood(t,1) = trapz(c_grid(:,t), f_cr(:,t));
    end
    % For diagnostics: compute implied mean and standard deviation of truncated normal
    % distribution.
    R_mean_truncnorm = R + sigma_r .* (normpdf(-R ./ sigma_r) ./ (1-normcdf(-R./sigma_r)));
    sigma_truncnorm = sqrt(sigma_r.^2 .* ( 1 + (-R./sigma_r .* normpdf(-R./sigma_r))./ (1-normcdf(-R./sigma_r)) ...
                                      - ( normpdf(-R./sigma_r)./(1-normcdf(-R./sigma_r))) .^2));
    % Compare expectation term to mean of truncated normal.
    R_diag = [R_mean_truncnorm, E_term_sum];
    % Check for validity of likelihood: not relevant for final, cleaned sample.
    n_ll_problematic = sum(likelihood==0);
    likelihood(likelihood==0) = 1E-8;
    fprintf('Number of non-defined likelihood points: %.0f \n', n_ll_problematic);
    ll = log(likelihood);
    % Final negative log-likelihood.
    neg_ll = - sum(log(likelihood));
end
