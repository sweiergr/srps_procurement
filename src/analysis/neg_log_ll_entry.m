function [neg_ll, log_likelihood] = neg_log_ll_entry(theta,y,X, N_pot)
%{

Return the negative of the log-likelihood evaluated at parameter vector
*theta* taking dependent variable *y* (actual observed number of entrants),
and regressor matrix *X* for the parameters, vector of number of bidders
to estimate probability of exactly *y* entrants entering the auction.
This lieklihood file can be used for both gross and net auctions. 

IMPORTANT 1: In this code N denotes the number of potential bidders, n is the number of actual bidders.
IMPORTANT 2: In this code, N and n denote the number of bidders INCLUDING the incumbent. Formula from paper for entrants have to be adjusted accordingly.

%}
    N_obs = size(X,1);
    % Initialize log likelihood and loop over auctions.
    likelihood = zeros(N_obs,1);
    log_likelihood = zeros(N_obs,1);
    for t=1:N_obs 
        % Compute likelihood contribution for one specific auction.
        n = y(t);
        N = N_pot(t);
        % Compute binomial coefficient to indicate total number of combinations.
        bin_coeff = nchoosek(N,n);
        % Compute probability of entrant entering the auction similar to ALS specification.
        q_num = exp(X(t,:) * theta);
        q = q_num ./ (1 + q_num);
        % Combine parts to get predited probability of n bidders entering.
        likelihood(t) = q.^(n) .* (1-q).^(N-n);
    end
    % Take logs and make likelihood negative since we want to minimize.    
    log_likelihood = log(likelihood);
    neg_ll = -sum(log_likelihood);
end