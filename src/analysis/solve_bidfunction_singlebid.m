function [bid_zero] = solve_bidfunction_singlebid(bid, signal_I,signal_E, N, lambda_I, lambda_E, rho_I, rho_E);
%{
    New bid function to solve numerically for given parameter values of
    bid function and a vector of (net) cost signals.

%}
    % Compute various CDFs and PDFs.
    CDF_I = wblcdf(bid(1),lambda_I,rho_I);
    PDF_I = wblpdf(bid(1),lambda_I, rho_I);
    CDF_E = wblcdf(bid(2),lambda_E,rho_E);
    PDF_E = wblpdf(bid(2),lambda_E,rho_E);
    % Compute markup terms for incumbent.
    G_MI = (1-CDF_E).^(N-1);
    g_MI = (N-1) .* (1-CDF_E).^(N-2) .* PDF_E;
    MU_I = G_MI ./ g_MI;
    
    % Compute markup for entrants.
    G_ME = (1-CDF_E).^(N-2) .* (1-CDF_I);
    % Workaround for ugly CDF_E values when number of bidders is only 2.
    if N==2
        g_ME = PDF_I .* (1-CDF_E).^ (N-2);
    else
        g_ME = (N-2) .* (1-CDF_E).^(N-3) .* PDF_E  .* (1-CDF_I) + PDF_I .* (1-CDF_E).^ (N-2);
    end
    MU_E = G_ME ./ g_ME;

    % Safety measure to avoid weird markup terms (after cleaning sample, these should not make a difference).
    MU_E(g_ME==0 & G_ME==0) = 0; % When other bidders bid lower for sure, no chance of winning, so charge zero markup (?)
    MU_I(g_MI==0 & G_MI==0) = 0; % 
    % Compute new bid for incumbent.
    bid_zero_I = signal_E + MU_I;
    bid_zero_E = signal_I + MU_E;
    % Stack bids for incumbent and entrant.
    bid_zero = [bid_zero_I;bid_zero_E] - bid;
end