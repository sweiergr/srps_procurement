function [parameter_matrix, lambda_vec_I, lambda_vec_E, rho_vec_I, rho_vec_E] = sim_wb_param(X_orig,theta)
    % Return a vector of scale and shape parameters of the Weibull distribution 
    % FOR BOTH INCUMBENT AND ENTRANT based on a linear index function of
    % regressor matrix *X* and paramter vector *theta* assuming a fully flexible % specification in which all paramters differ across incumbent and entrants.
    global N_obs
    K = size(X_orig,2);
    % Expand regressor matrix.
    X = repmat(X_orig,1,4); 
    % Initialize grid for parameter values.
    % Stack in columns: lambda_I, lambda_E, rho_I, rho_E
    parameter_matrix = zeros(N_obs,4);
    for  i=1:4
        parameter_matrix(:,i) = exp(X(:,((i-1)*K)+1:i*K) * theta(((i-1)*K)+1:i*K));
    end
    lambda_vec_I = parameter_matrix(:,1);
    rho_vec_I = parameter_matrix(:,3);
    lambda_vec_E = parameter_matrix(:,2);
    rho_vec_E = parameter_matrix(:,4);
end