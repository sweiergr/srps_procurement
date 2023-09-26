function [grad_matrix] = num_gradient_2(theta, min_neg_log_ll,  tol, speed)
% Calculate numerical derivatives for gradient of the log-likelihood specified 
% in *ll_type*.
    
    global N_obs
    K = length(theta);
    grad = zeros(K,1);
    % Evaluate E(log(f_i)) at theta_opt.
    [log_ll_opt, ll_opt] = min_neg_log_ll(theta);
    
    % Create derivative matrix for every observation and coefficient.
    f_prime_new = zeros(K,N_obs);
    f_prime_old = zeros(K,N_obs);
    
    % Iterate over components of theta.
    for j=1:K
        delta = (1:K==j)';
        diff=1;
        m = 1;
        
        % Check which step size works well here.
        step_size=1E-13;
        % Multiplicative perturbation.
        theta_grad_fwd = theta + delta .* step_size .* theta ;
        theta_grad_bwd = theta - delta .* step_size .* theta;
        [log_opt_grad_fwd,ll_grad_fwd] = min_neg_log_ll(theta_grad_fwd);
        [log_opt_grad_bwd,ll_grad_bwd] = min_neg_log_ll(theta_grad_bwd);
        f_prime_new(j,:) = inv(2.* step_size .* theta(j,1)) .* ( ll_grad_fwd - ll_grad_bwd);
        % SW: For us, multiplicative perturbation was much more stable than additive.
        % Additive perturbation.
%         theta_grad_fwd = theta + delta .* step_size;
%         theta_grad_bwd = theta - delta .* step_size;
%         [log_opt_grad_fwd,ll_grad_fwd] = min_neg_log_ll(theta_grad_fwd);
%         [log_opt_grad_bwd,ll_grad_bwd] = min_neg_log_ll(theta_grad_bwd);
%         f_prime_new(j,:) = inv(2 .* step_size) .* ( ll_grad_fwd - ll_grad_bwd);
    end
    
    grad_matrix = (f_prime_new * f_prime_new') ./ (N_obs);
end

