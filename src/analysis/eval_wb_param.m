function [parameter_matrix, lambda_vec, rho_vec] = eval_wb_param(X,theta)
    % Return a vector of scale and shape parameters of the Weibull
    % distribution based on a linear index function of regressor matrix *X*
    % and paramter vector *theta* assuming a fully flexible specification
    % in which all apramters differ across incumbent and entrants.
    global N_obs
    % Create number of acutally used regressors.
    % Assuming fully flexible model with 4 parameters per regressor is
    % estimated.
    K_eval = size(X,2) ./ 4;
    % Initialize grid for paramter values.
    % Stack in columns: lambda_I, lambda_E, rho_I, rho_E
    parameter_matrix = zeros(N_obs,2);
    for  i=1:2
        parameter_matrix(:,i) = exp(X(:,((i-1)*2*K_eval)+1:i*2*K_eval) * theta(((i-1)*2*K_eval)+1:i*2*K_eval));
    end
    lambda_vec = parameter_matrix(:,1);
    rho_vec = parameter_matrix(:,2);
end