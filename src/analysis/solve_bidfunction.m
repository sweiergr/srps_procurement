function [bid_zero] = solve_bidfunction(bid_grid, signal_I, signal_E, N, lambda_I, lambda_E, rho_I, rho_E);
%{
    Solve for a new bid function numerically for given parameter values of
    bid function and a vector of (net) cost signals.

%}
    bidgrid_size = length(bid_grid) ./ 2;
    % Compute various CDFs and PDFs.
    CDF_I = wblcdf(bid_grid(1:bidgrid_size),lambda_I,rho_I);
    PDF_I = wblpdf(bid_grid(1:bidgrid_size),lambda_I, rho_I);
    CDF_E = wblcdf(bid_grid(bidgrid_size+1:end),lambda_E,rho_E);
    PDF_E = wblpdf(bid_grid(bidgrid_size+1:end),lambda_E, rho_E);
    % Compute markup terms for incumbent.
    G_MI = (1-CDF_E).^(N-1);
    g_MI = (N-1) .* (1-CDF_E).^(N-2) .* PDF_E;
    MU_I = G_MI ./ g_MI;
    % Compute markup for entrants.
    G_ME = (1-CDF_E).^(N-2) .* (1-CDF_I);
    g_ME = (N-2) .* (1-CDF_E).^(N-3) .* PDF_E  .* (1-CDF_I) + PDF_I .* (1-CDF_E).^ (N-2);
    MU_E = G_ME ./ g_ME;
    % Safety measure to avoid weird markup terms.
    MU_E(g_ME==0 & G_ME==0) = 0;
    MU_I(g_MI==0 & G_MI==0) = 0;
    MU_E(g_ME==0 & G_ME==1) = 50;
    MU_I(g_MI==0 & G_MI==1) = 50;
    % Compute new bid for incumbent.
    bid_zero_I = signal_E + MU_I;
    bid_zero_E = signal_I + MU_E;
    % Stack bids for incumbent and entrant.
    bid_zero = [bid_zero_I;bid_zero_E] - bid_grid;
end