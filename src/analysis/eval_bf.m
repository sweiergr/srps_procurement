function [bid_CDF, bid_PDF] = eval_bf(lambda,rho,bid,trunc_point)
    % Return the CDF and PDF of estimated bid functions based on scale and shape
    % parameters and a given bid vector.
    % Set default truncation bid at 0.
    if nargin==3
        trunc_point=0;
    end
    % Compute some subparts of the likelihood.
    exp_B = exp(-(bid ./ lambda).^rho);
    % Compute CDF of bid.
    bid_CDF = 1 - exp_B;
    % Compute density of bid.
    bid_PDF = exp_B .* (bid ./ lambda).^(rho-1) .* rho./lambda;
    % Compute probability mass below truncation point.
    mass_truncated = 1 - wblcdf(trunc_point,lambda,rho);
    % Compute CDF value at truncation point.
    CDF_trunc_bid = wblcdf(trunc_point, lambda,rho);
    % Scale up truncated PDF.
    bid_PDF = bid_PDF ./ mass_truncated;
    % Scale up truncated CDF.
    bid_CDF = (bid_CDF - CDF_trunc_bid) ./ mass_truncated;
end