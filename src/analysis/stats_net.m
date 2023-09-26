function [std_errors, t_stats, p_values] = stats_net(theta, min_neg_log_ll, tol, speed)
% Return standard errors based on numerical gradients of the *theta*
% estimates. In addition return t-statistics and p-values.
% *ll_type* specifies the type of likelihood, *tol* and *speed* are
% convergence options used in *num_gradient.m* to calculate numerical
% values of the gradient.
    global N_obs
    K = length(theta);
    % Calculation of standard errors.    
    % Calculate gradient of log-likelihood at optimal theta-values.
    grad_matrix = num_gradient_net(theta, min_neg_log_ll, tol, speed);
    % Calculate asymptotic variance-covariance matrix.
    asy_vcov = inv(grad_matrix);
    %disp('Results for standard error estimates:');
    std_errors = sqrt(diag(asy_vcov) ./ N_obs);
    % Calculate t-statistics.
    t_stats = theta ./ std_errors;
    % Calculate asymptotic p-values.
    p_values = 2 * (1 - normcdf(abs(t_stats),0,1)); 
end