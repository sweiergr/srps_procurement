function [bid_CDF, bid_PDF] = eval_bid_function(lambda,rho,bid)
    % Return the CDF and PDF of estimated bid functions based on scale and shape
    % parameters and a given bid.
    % Compute density of winning bid.
    % Compute some subparts of the likelihood.
    exp_B = exp(-(bid ./ lambda).^rho);
    bid_CDF = 1 - exp_B;
    bid_PDF = exp_B .* (bid ./ lambda).^(rho-1).*rho./lambda;
end