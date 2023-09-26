function X_IE_solve = solve_E_FP(X_IE, alpha_E, alpha_I, R, sigma_r, RHO, C, N, db_prob)
%{
This function computes the fixed point for the evaluation of the
expectation terms in the net auctions. The function takes scalar
arguments for one auction and outputs a pair of expectation terms (for
incumbent and entrants).
    Input arguments:
    - alpha_E, alpha_I: asymmetry parameters
    - R: expected value of revenue distribution (i.e. of parent normal
    distribution)
    - sigma_r: standard deviation of revenue distribution (i.e. of parent
    normal distribution)
    - RHO: vector of total net cost signals from first-step
    - C: container for cost densities for both incumbent and entrant
    - N: number of bidders in auction
    - db_prob: Conditional probability of DB being pivotal bidder.
    
    % Order of terms in fixed point search:
    % 1. X_I_eq
    % 2. X_E_eq
    % 3. X_I_plus
    % 4. X_E_plus

%}
    % Container for grid points.
    gp_kdens = size(C,1);
    % Expand variables for cost integration: RHO-signals
    RHO_E = repmat(RHO(2),gp_kdens,1);
    RHO_I = repmat(RHO(1),gp_kdens,1);
    % Extract cost grid for entrants and incumbent.
    c_grid_E = C(:,:,2,2);
    c_grid_I = C(:,:,1,2);
    % Diagnostics: check what happens if we shift cost distribution to
    % the right.
    % c_grid_E = c_grid_E + 15;
    % c_grid_I = c_grid_I + 15;
    % Extract cost density for entrants and incumbent.
    f_c_E = C(:,:,2,1);
    f_c_I = C(:,:,1,1);

    % Ensure that we only search over positive values: typically not
    % needed.
    % Should we restrict X_IE to be positive using an exponential
    % transformation? Typically, we don't get negative solutions anyway.
    % X_IE = exp(X_IE);
    
    % Compute argument of cost integral, i.e. revenue signal as function of
    % RHO, c and parameters of model.
    if N==2 % case distinction only relevant for r_E.
        arg_E = (c_grid_E - RHO_E - alpha_I .* repmat(X_IE(1),gp_kdens,1)) ./alpha_E;
    elseif N>2
        arg_E = (c_grid_E - RHO_E - ...
                db_prob .* ( (N-2) .* alpha_E .* repmat(X_IE(4),gp_kdens,1) + alpha_I .* repmat(X_IE(1),gp_kdens,1)) - ...
                (1-db_prob) .* ((N-3) .* alpha_E .* repmat(X_IE(4),gp_kdens,1) + alpha_I .* repmat(X_IE(3),gp_kdens,1) + alpha_E .* repmat(X_IE(2),gp_kdens,1)) ...
                ) ./alpha_E;
    end
    arg_I = (c_grid_I - RHO_I ...
             - alpha_E .* repmat(X_IE(2),gp_kdens,1) ...
             - (N-2) .* alpha_E .* repmat(X_IE(4),gp_kdens,1)) ./alpha_I;
    
    % Compute density of revenue signal with r following truncated normal.
    % Use symmetry of normal CDF to save a few calculations.
    truncated_normpdf_I = normpdf((arg_I-R)./sigma_r) ./ sigma_r ./ normcdf(R ./ sigma_r);
    truncated_normpdf_E = normpdf((arg_E-R)./sigma_r) ./ sigma_r ./ normcdf(R ./ sigma_r);
        
    % Diagnostics: check mean of revenue density.
    % mean_density = mean([truncated_normpdf_I, truncated_normpdf_E]);
    % Set density for negative arguments to zero.
    f_r_E = (arg_E >=0) .* truncated_normpdf_E + 0 .* (arg_E < 0) ;
    f_r_I = (arg_I >=0) .* truncated_normpdf_I + 0 .* (arg_I < 0) ;

    % DIAGNOSTICS: Check reasonability of cost distribution.
    % Check whether distributions integrate to one.
    % Looks like generally cost distributions are ok.
    int_cost_I = trapz(c_grid_I,f_c_I);
    int_cost_E = trapz(c_grid_E,f_c_E);
    
    % Integrate over joint density.
    % This computes the total probability mass consistent with RHO.
    % Please check that this is correct!
    int_jointdens_I = trapz(c_grid_I,f_c_I.*f_r_I);
    int_jointdens_E = trapz(c_grid_E,f_c_E.*f_r_E);
    
    % Safety measure: for debugging when some lines result in only negative
    % r-signals.
    int_jointdens_I(int_jointdens_I==0) = 1E-8;
    int_jointdens_E(int_jointdens_E==0) = 1E-8;
    
    % Compute integral over joint density corrected with conditoning integral above.
    % This should integrate to one by construction, looks generally good.
    int_joint_correct_I = trapz(c_grid_I,f_c_I.*f_r_I./int_jointdens_I);
    int_joint_correct_E = trapz(c_grid_E,f_c_E.*f_r_E./int_jointdens_E);
    
    % Compute density of convolution.
    joint_dens_I = f_c_I .* f_r_I; 
    joint_dens_E = f_c_E .* f_r_E;
    % Integrate over r-c-signals using convolution for given X_IE.
    cost_int_I = trapz(c_grid_I, arg_I .* joint_dens_I);
    cost_int_E = trapz(c_grid_E, arg_E .* joint_dens_E);
    % Construct system of equations to solve for fixed point.
    X_I_solve = X_IE(1) - cost_int_I;
    X_E_solve = X_IE(2) - cost_int_E;
    
    %% Compute terms for incumbent and entrants analogously.
    % Define some common parameters and objects used for both entrants and
    % DB. Again, exploiting symmetry of normal CDF...
    cond_r_dens_den = normcdf(R./sigma_r); % denominator for truncated CDF for conditioning revenue distribution.
    n_r_grid = 250; % number of grid points for numerical integration of revenue distribution.
    r_dens_pdf_den = sigma_r .* normcdf(R./sigma_r); % denominator of trunacted normal pdf.
    
    % Upper bounds on revenue signal.
    r_I_ub = arg_I;
    r_E_ub = arg_E;
    % Compute conditioning denominator of revenue distribution.
    % This comes from the revenue CDFs evaluated at upper bounds.
    cond_r_I_dens_num = max(normcdf( (r_I_ub-R) ./ sigma_r) - normcdf(-R./sigma_r),0);
    cond_r_I_dens = cond_r_I_dens_num ./ cond_r_dens_den;
    cond_r_E_dens_num = max(normcdf( (r_E_ub-R) ./ sigma_r) - normcdf(-R./sigma_r),0);
    cond_r_E_dens = cond_r_E_dens_num ./ cond_r_dens_den;
    
    % Define step size grid for revenue grids.
    % This is a matrix-compatible alternative to linspace-function and
    % avoids having to loop over cost realizations.
    stepsizes_I = r_I_ub/(n_r_grid-1);
    stepsizes_E = r_E_ub/(n_r_grid-1);

    % Create revenue grids for each revenue threshold.
    % Careful here: In case upper revenue bound is negative, this will go
    % backwards from zero to negative number. If model is well-behaved, this should not happen.
    r_I_grid = (0:(n_r_grid-1)).*stepsizes_I;
    r_E_grid = (0:(n_r_grid-1)).*stepsizes_E;

    % Compute unconditional revenue densities.
    % Careful: this conditioning density can occasionally be zero. Need to
    % deal with this as special case below: Set total integral to zero in this case!
    r_I_dens = normpdf((r_I_grid-R) ./ sigma_r) ./ r_dens_pdf_den;
    r_E_dens = normpdf((r_E_grid-R) ./ sigma_r) ./ r_dens_pdf_den;
    % Safety measure: density at negative revenue signals is set to zero.
    r_I_dens(r_I_grid<=0) = 0;
    r_E_dens(r_E_grid<=0) = 0;
    
    % Given values for density and grid for r values, approximation to
    % integral is just sample average over the grid points.
    % Gives expected value of revenue signal for a given cost realization.
    r_I_exp = mean(r_I_grid .* r_I_dens,2) ./ cond_r_I_dens;
    r_E_exp = mean(r_E_grid .* r_E_dens,2) ./ cond_r_E_dens;
    % Safety measure against NaNs due to impossible cost-revenue signal
    % combinations. 
    r_I_exp(isnan(r_I_exp))=0;
    r_E_exp(isnan(r_E_exp))=0;
    % Finally, integrate over cost distribution.
    cost_int_I_plus = trapz(c_grid_I, r_I_exp .* f_c_I);
    cost_int_E_plus = trapz(c_grid_E, r_E_exp .* f_c_E);
    % Arrange plus-terms for fsolve.
    X_I_plus = X_IE(3) - cost_int_I_plus;
    X_E_plus = X_IE(4) - cost_int_E_plus;
    % Combine the four relevant terms to solve for fixed point.
    X_IE_solve = [X_I_solve; X_E_solve;X_I_plus;X_E_plus];
end