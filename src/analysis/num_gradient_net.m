function [grad_matrix] = num_gradient_net(theta, min_neg_log_ll,  tol, speed)
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
        
        % Compute derivative using multiplicative two-sided perturbation.
        % step = 0.02;
        % theta_grad_plus = theta + delta .* step .* theta ;
        % theta_grad_minus = theta - delta .* step .* theta ;
        % [log_opt_grad_plus,ll_grad_plus] = min_neg_log_ll(theta_grad_plus);
        % [log_opt_grad_minus,ll_grad_minus] = min_neg_log_ll(theta_grad_minus);
        % % Based on only forward perturbation.
        % f_prime_new(j,:) = inv(step .* theta(j,1)) .* ( ll_grad_plus - ll_opt);

        % Compute gradient based on backward and forward perturbation.
        % Check how sensitive gradient is to step size.
        step = 0.015;
        theta_grad_plus = theta + delta .* step .* theta;
        theta_grad_minus = theta - delta .* step .* theta;
        [log_opt_grad_plus,ll_grad_plus] = min_neg_log_ll(theta_grad_plus);
        [log_opt_grad_minus,ll_grad_minus] = min_neg_log_ll(theta_grad_minus);
        f_prime_new(j,:) = inv(2 .* step .* theta(j,1)) .* ( ll_grad_plus - ll_grad_minus);
    end
    grad_matrix = (f_prime_new * f_prime_new') ./ N_obs;
end

