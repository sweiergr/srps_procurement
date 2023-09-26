function [grad_matrix] = num_gradient_gross(theta, min_neg_log_ll,  tol, speed)
% Calculate numerical derivatives for gradient of the log-likelihood.
    
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

        % Compute numerical derivative using two-sided perturbation.
        step = 0.0025;
        theta_grad_plus = theta + delta .* step .* theta ;
        theta_grad_minus = theta - delta .* step .* theta ;
        [log_opt_grad_plus,ll_grad_plus] = min_neg_log_ll(theta_grad_plus);
        [log_opt_grad_minus,ll_grad_minus] = min_neg_log_ll(theta_grad_minus);
        % Based on only forward perturbation.
        f_prime_new(j,:) = inv(step .* theta(j,1)) .* ( ll_grad_plus - ll_opt);

%         % Compute gradient based on absolute change in theta.
%         % SW: Additive didn't work as well as multiplicative here.
%         step = 0.1;
%         theta_grad_plus = theta + delta .* step;
%         theta_grad_minus = theta - delta .* step;
%         % TO ADD: PERTURB FORWARD AND BACKWARD.
%         %theta_grad = theta + 0.01;
%         [log_opt_grad_plus,ll_grad_plus] = min_neg_log_ll(theta_grad_plus);
%         [log_opt_grad_minus,ll_grad_minus] = min_neg_log_ll(theta_grad_minus);
%         f_prime_new(j,:) = inv(2 .* step) .* ( ll_grad_plus - ll_grad_minus);
%         f_prime_new(j,:) = inv(0.01) .* ( ll_grad - ll_opt);
    end
    grad_matrix = (f_prime_new * f_prime_new') ./ N_obs;
end

