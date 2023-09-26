function X_IE_solve = solve_E_FP(X_IE, alpha_E, alpha_I, R, sigma_r, RHO, C, N)
% This function computes the fixed point for the evaluation of the
% expectation terms in the net auctions. The function takes scalar
% arguments for one auction and outputs a pair of expectation terms (for
% incumbent and entrants).

    gp_kdens = size(C,1);
    % Compute integral over costs.
    % Expand variables for cost integration: RHO-signals
    RHO_E = repmat(RHO(2),gp_kdens,1);
    RHO_I = repmat(RHO(1),gp_kdens,1);
    % Extract cost grid for entrants and incumbent.
    c_grid_E = C(:,:,2,2);
    c_grid_I = C(:,:,1,2);
    % Extract cost density for entrants and incumbent.
    f_c_E = C(:,:,2,1);
    f_c_I = C(:,:,1,1);

    % Ensure that we only search over positive values.
    % Should we restrict X_IE to be positive using an exponential
    % transformation? Typically, we don't get negativ solutions anyway.
    % X_IE = exp(X_IE);
    % Compute argument of cost integral.
    arg_E = (c_grid_E - RHO_E - (N-2) .* alpha_E .* repmat(X_IE(2),gp_kdens,1) - alpha_I .* repmat(X_IE(1),gp_kdens,1)) ./alpha_E;
    arg_I = (c_grid_I - RHO_I - (N-1) .* alpha_E .* repmat(X_IE(2),gp_kdens,1)) ./alpha_I;

    % Compute density of revenue signal with r following truncated normal.
    % Compute density for positive arguments.
    truncated_normpdf_I = normpdf((arg_I-R)./sigma_r) ./ sigma_r ./ normcdf(R ./ sigma_r);
    truncated_normpdf_E = normpdf((arg_E-R)./sigma_r) ./ sigma_r ./ normcdf(R ./ sigma_r);
    % Set density for negative arguments to zero.
    f_r_E = (arg_E >=0) .* truncated_normpdf_E + 0 .* (arg_E < 0) ;
    f_r_I = (arg_I >=0) .* truncated_normpdf_I + 0 .* (arg_I < 0) ;

    % Cost distribution and integration when using unconditional cost
    % distribution.
    cost_int_E = trapz(c_grid_E, arg_E .* f_r_E .* f_c_E);
    cost_int_I = trapz(c_grid_I, arg_I .* f_c_I .* f_c_I);
   
    %% Alternative?
    % Key question: Should we condition the cost distribution?
    % What does a negative argument in revenue distribution mean? Not
    % reconcilable with a given cost realization. Therefore, use:
    f_c_E_cond = f_c_E;
    f_c_I_cond = f_c_I;
    f_c_E_cond(arg_E<0) = 0;
    f_c_I_cond(arg_I<0) = 0;
    mass_ex_cost_E = sum(f_c_E(arg_E<0)) ./ sum(f_c_E);
    mass_ex_cost_I = sum(f_c_I(arg_I<0)) ./ sum(f_c_I);
    % Is this conditioning done correctly?
    f_c_E_cond(arg_E>=0) = f_c_E(arg_E>0) ./ (1-mass_ex_cost_E);
    f_c_I_cond(arg_I>=0) = f_c_I(arg_I>0) ./ (1-mass_ex_cost_I);
    % Just for double-checking how big the difference is between
    % unconditional and conditional cost distribution.
    %sum(arg_E<0);
    %sum(arg_I<0);
    
    % Integration using conditional cost distribution.
    %cost_int_E = trapz(c_grid_E, arg_E .* f_r_E .* f_c_E_cond);
    %cost_int_I = trapz(c_grid_I, arg_I .* f_r_I .* f_c_I_cond);
  
    % Construct system of equations to solve for fixed point.
    X_I_solve = X_IE(1) - cost_int_I;
    X_E_solve = X_IE(2) - cost_int_E;
    X_IE_solve = [X_I_solve; X_E_solve];
end