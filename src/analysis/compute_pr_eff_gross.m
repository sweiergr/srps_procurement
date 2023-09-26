%{
    Counterfactual preparation for gross auction sample.
    Compute probability of choosing efficient firm in gross auctions and expected subsidies and agency payoffs.
    
%}

clear
clc
format('short');
% Define necessary globals.
global N_obs K
% Incumbent in first column, entrant in second column.
clf('reset')

% Indicate what to do.
update_plots = 1;
% Prevent graphs from popping up.
set(gcf,'Visible', 'on'); 
% SET PAPER SIZE FOR FIGURES TO AVOID LARGE MARGINS.
figuresize(15,12,'cm')

% Set seed in case any simulation is used.
rng(123456);

%% Load gross auction workspace and net auction parameters.
load(project_paths('OUT_ANALYSIS','grossauction_workspace'));
T = length(db_win);
% Setting options for fsolve to get cleaner display output.
options_fsolve = optimoptions('fsolve','Display','none');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load probabilities of different bidder configurations in gross auctions.
load(project_paths('OUT_ANALYSIS','ga_entry_npot'));
N_pot_max = max(N_pot_gross);
load(project_paths('OUT_ANALYSIS','ga_entry_n_probs'));
prob_N_grid = prob_N_grid_gross;
% Check whether probability of n bidders entering sum up to one.
check_prob_gross_N = sum(prob_N_grid,2);
prob_no_entrant = 1 - check_prob_gross_N;

% Analyze how number of bidders is likely to change.
% It's not clear whether endogenous N goes up or down compared to observed N.
comp_N = [N-1, prob_N_grid];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialize containers for efficiency and subsidy numbers.
size_bidgrid = 500;
% Container for final probabilities of interest.
Pr_efficient_gross = zeros(T,1);
% Alternative version in which we allow for no entrant entering as possibility.
Pr_efficient_gross_alt = zeros(T,1);
sum_Pr_win_gross = zeros(T,N_pot_max);
Pr_efficient_gross_constant_n = zeros(T,1);
sum_Pr_win_gross_constant_n = zeros(T,1);
% Expected subsidy.
E_subsidy_gross = zeros(T,1);
E_subsidy_gross_constant_n = zeros(T,1);
% Alternative version in which we allow for no entrant entering as possibility.
E_subsidy_gross_alt = zeros(T,1);
% Container for expected bid for each auction and potential bidder configuration.
E_bid_endo_n = zeros(T,N_pot_max);
% Container for efficiency probability for each auction and potential bidder configuration.
Pr_eff_endo_n = zeros(T,N_pot_max);
% Create containers for case when no entrant enters.
% By construction efficiency probability is one. 
% For revenue, we can compute the expected bid based on the bid distribution extrapolated to N=1.
E_bid_zeroN = zeros(T,1);

% Absolute max of bid grid to analyze (50% over maximum bid observed in
% the data).
bid_win_global_max = 5 * max(bid_win);
% Absolute minimum of bid grid to analyze (for now set to zero.
bid_win_global_min = 0.1 * min(bid_win);

% Create indicator matrix for whether auction is "well-behaved" or either
% entrant or incumbent have dominant bid distribution.
% 99: regular auction with neither bidder type dominating.
% 5500: entrant is dominant in auction-bidder configuration combination.
% 1100: incumbent is dominant in auction-bidder configuration combination.
% any negative number: both bidder types are dominant (this should not happen, investigate
% if this pops up!)
auction_indicator = 99 .* ones(T,N_pot_max);

% Loop over auctions.
for t=1:T
    % This is the maximum TOTAL number of bidders that we could observe in
    % the auction based on our data.
    N_max_t = N_pot_gross(t);
    % Prepare objects for computation of bid function parameters.
    X_aux_endo_n = X_aux(t,:);
    
    % Truncate simulated bids at 10 times observed winning bid to avoid
    % extreme outliers.
    bid_win_global_max = 100 * max(bid_win(t));
    % Truncate simulated bids below at 10% of observed winning bid.
    bid_win_global_min = 0.05 * min(bid_win(t));

    %% Compute expected bid when no entrant enters.
    % Recall that in bid function, parameters use min(n,5).
    X_aux_endo_n(end) = log(min(0+1,5)) ./ 10;
    % Compute new bid function parameters assuming n_hyp entrant
    % bidders.
    lambda_I_n = exp(X_aux_endo_n * theta_opt(1:K)); 
    lambda_E_n = exp(X_aux_endo_n * theta_opt(K+1:2*K));
    rho_I_n = exp(X_aux_endo_n * theta_opt(2*K+1:3*K));
    rho_E_n = exp(X_aux_endo_n * theta_opt(3*K+1:4*K));
    % Set up an auction-specific bid grid.
    bid_min_I = wblinv(0.005,lambda_I_n,rho_I_n);
    bid_min_E = wblinv(0.005,lambda_E_n,rho_E_n);
    bid_max_I = wblinv(0.995,lambda_I_n,rho_I_n);
    bid_max_E = wblinv(0.995,lambda_E_n,rho_E_n);
    % Expansive bid grid that covers both bid ranges.
    bg_min_large = max(min(bid_min_I,bid_min_E),bid_win_global_min);
    bg_max_large = min(max(bid_max_I,bid_max_E),bid_win_global_max);
    % Minimal bid grid on area only where both bid distributions overlap.
    bg_min_small = max(max(bid_min_I,bid_min_E),bid_win_global_min);
    bg_max_small = min(min(bid_max_I,bid_max_E),bid_win_global_max);
    
    % Minimal grid that covers only range where both bid distributions
    % overlap.
    if bg_min_small >= bg_max_small
        bid_grid_small = 0;
    else
        bid_grid_small=linspace(bg_min_small,bg_max_small,size_bidgrid)';
    end
    % Extensive bid grid covering both type's bid distributions.
    bid_grid_large=linspace(bg_min_large,bg_max_large,size_bidgrid)';
    
    % Compute various CDFs and PDFs.
    CDF_E = wblcdf(bid_grid_large,lambda_E_n,rho_E_n);
    PDF_E = wblpdf(bid_grid_large,lambda_E_n, rho_E_n);
    CDF_I = wblcdf(bid_grid_large,lambda_I_n,rho_I_n);
    PDF_I = wblpdf(bid_grid_large,lambda_I_n, rho_I_n);
    E_bid_zeroN(t,1)  = trapz(bid_grid_large,bid_grid_large .* PDF_I);
            
%% end computation of bid when no entrant enters.

    %% LOOP OVER ALL POTENTIAL N when at least one bidder enters.
    % n_hyp indicates the total number of entrants entering the auction.
    fprintf('Analyzing gross auction %d, which has %d entrants observed entering...\n',t,N(t));

    for n_hyp=1:N_max_t
        % Recall that in bid function, parameters use min(n,5).
        X_aux_endo_n(end) = log(min(n_hyp+1,5)) ./ 10;
        % Compute new bid function parameters assuming n_hyp entrant
        % bidders.
        lambda_I_n = exp(X_aux_endo_n * theta_opt(1:K)); 
        lambda_E_n = exp(X_aux_endo_n * theta_opt(K+1:2*K));
        rho_I_n = exp(X_aux_endo_n * theta_opt(2*K+1:3*K));
        rho_E_n = exp(X_aux_endo_n * theta_opt(3*K+1:4*K));

        % Set up an auction-specific bid grid.
        bid_min_I = wblinv(0.001,lambda_I_n,rho_I_n);
        bid_min_E = wblinv(0.001,lambda_E_n,rho_E_n);
        bid_max_I = wblinv(0.999,lambda_I_n,rho_I_n);
        bid_max_E = wblinv(0.999,lambda_E_n,rho_E_n);
        
        % Expansive bid grid that covers both bid ranges.
        bg_min_large = max(min(bid_min_I,bid_min_E),bid_win_global_min);
        bg_max_large = min(max(bid_max_I,bid_max_E),bid_win_global_max);
        % Minimal bid grid on area only where both bid distributions overlap.
        bg_min_small = max(max(bid_min_I,bid_min_E),bid_win_global_min);
        bg_max_small = min(min(bid_max_I,bid_max_E),bid_win_global_max);
    
        % Minimal grid that covers only range where both bid distributions
        % overlap.
        if bg_min_small >= bg_max_small
            bid_grid_small = 0;
        else
            bid_grid_small=linspace(bg_min_small,bg_max_small,size_bidgrid)';
        end
        % Extensive bid grid covering both type's bid distributions.
        bid_grid_large=linspace(bg_min_large,bg_max_large,size_bidgrid)';
            
        % Debug command, clean up later.
        bid_grid = bid_grid_large;
        
        % Test whether one bidder type dominates the other.
        dominant_entrant = (bid_max_E < bid_min_I);
        dominant_incumbent = (bid_max_I < bid_min_E);
        auction_indicator(t,n_hyp) = (auction_indicator(t,n_hyp) - 44 .* dominant_entrant);
        auction_indicator(t,n_hyp) = (auction_indicator(t,n_hyp) - 88 .* dominant_incumbent);
        
        % Compute various CDFs and PDFs.
        CDF_E = wblcdf(bid_grid,lambda_E_n,rho_E_n);
        PDF_E = wblpdf(bid_grid,lambda_E_n, rho_E_n);
        CDF_I = wblcdf(bid_grid,lambda_I_n,rho_I_n);
        PDF_I = wblpdf(bid_grid,lambda_I_n, rho_I_n);
        % Compute probabilities of incumbent and entrant winning for bid_grid.
        P_I_win = PDF_I .* (1-CDF_E).^(n_hyp);
        P_E_win = (n_hyp) .* PDF_E .* (1-CDF_E).^(n_hyp-1) .* (1-CDF_I);
        % Check that winning probabilities add up to one.
        sum_Pr_win_gross(t,n_hyp) = trapz(bid_grid_large,P_E_win) + trapz(bid_grid_large,P_I_win);
        
        % Compute markup terms for incumbent.
        G_MI = (1-CDF_E).^(n_hyp); % Pr that all entrants bid more.
        g_MI = (n_hyp) .* (1-CDF_E).^(n_hyp-1) .* PDF_E; % derivative of Pr that all entrants bid more.
        MU_I = G_MI ./ g_MI;
        
        % Compute markup for entrants.
        G_ME = (1-CDF_E).^(n_hyp-1) .* (1-CDF_I); % Pr that all rival entrants and incumbent bid more.
        % This should not do anything, but just to be safe: avoid weird markup behavior for N=2.
        if n_hyp==1
            g_ME = PDF_I .* (1-CDF_E);
        else
            g_ME = (n_hyp-1) .* (1-CDF_E).^(n_hyp-2) .* PDF_E  .* (1-CDF_I) + PDF_I .* (1-CDF_E).^(n_hyp-1);
        end
        MU_E = G_ME ./ g_ME;
        
        % Compute expected subsidy paid by the agency.
        E_bid_endo_n(t,n_hyp) = trapz(bid_grid,bid_grid .* (PDF_I .* G_MI + (n_hyp) .* PDF_E .* G_ME));
        % Check how many cost draws are negative.
        % sprintf('Number of negative signals for incumbent:')
        % sum(signal_I<0)
        % Check how many cost draws are negative.
        % sprintf('Number of negative signalss for entrant:')
        % sum(signal_E<0)
        % END OF COMPUTATION OF COUNTERFACTUAL BID.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %% Special case of markups when entrants bid distribution is completely to left of incumbents.
        % Entrant always wins:
        % We assume that the following happens:
        % Incumbent always bids zero markups because never wins the
        % auction.
        % Entrant always wins, not clear what markup it should charge.
        % For now we assume that it bids such that it always wins, i.e., it
        % bids:
        % Test where entrant is dominant in this auction-bidder
        % configuration.
        % Define de facto dominance if 0.999-percentile of entrants
        % distribution is below incumbents 0.001-percentile.
        if dominant_entrant==1 % entrant always wins the auction.
            fprintf('Calculating for gross auction %d and %d entrants entering...\n',t,n_hyp);
            fprintf('Entrant has dominant bid distribution, efficiency prob purely based on cost distributions!\n');
            % Entrant always wins, so efficiency probability is probability
            % the the lowest cost entrant has lower cost than the incumbent
            c_grid = kdens_I_grid(t,:);
            pdf_c_i = kdens_I(t,:);
            cdf_c_e = kCDF_E(t,:);
            % Compute probability that incumbent has a cost lower than the
            % lowest cost entrant. This is the probability of having an
            % inefficient auction.
            RR_ineff_E = pdf_c_i .* (1.0-cdf_c_e).^n_hyp;
            Pr_e_ineff = trapz(c_grid,RR_ineff_E);
            % Write efficiency probability to container.
            Pr_eff_endo_n(t,n_hyp) = 1.0 - Pr_e_ineff;
            fprintf('Pr that winning entrant is efficient (based on cost distribution): %6.4f.\n',Pr_eff_endo_n(t,n_hyp));
        elseif dominant_incumbent==1 % incumbent always wins the auction.
            fprintf('Calculating for gross auction %d and %d entrants entering...\n',t,n_hyp);
            fprintf('Incumbent has dominant bid distribution, efficiency prob purely based on cost distributions!\n');
            % Incumbent always wins, so efficiency probability is probability
            % the the lowest cost entrant has higher cost than the
            % incumbent.
            c_grid = kdens_I_grid(t,:);
            pdf_c_e = kdens_E(t,:);
            cdf_c_i = kCDF_I(t,:);
            % Compute probability that incumbent has a cost lower than the
            % lowest cost entrant. This is the probability of having an
            % efficient auction.
            RR_eff_I = pdf_c_i .* (1.0-cdf_c_e).^n_hyp;
            Pr_i_eff = trapz(c_grid,RR_eff_I);
            % Write efficiency probability to container.
            Pr_eff_endo_n(t,n_hyp) = Pr_i_eff;
            fprintf('Pr that winning incumbent is efficient (based on cost distribution): %6.4f.\n',Pr_eff_endo_n(t,n_hyp));
        else % "normal" case where both bidder types could win.
            
            % Compute vector of incumbent's costs.
            signal_I = bid_grid - MU_I;
            % Compute vector of entrants' costs.
            signal_E = bid_grid - MU_E;
            plot(bid_grid,signal_E,bid_grid,signal_I);
            legend('Entrant','Incumbent');
            title('Evolution of signals over bid grid');
            
            % Compute bid at other type's signal.
            % Solve new system of bidding FOCs numerically.
            % Starting values for system of solving for bid functions.
            new_bid = zeros(2,size_bidgrid);
            % Solve for new bids.
            for bn=1:length(bid_grid)
                single_bid = repmat(bid_grid(bn),2,1);
                solve_hyp_bf_singlebid = @(bid) solve_bidfunction_singlebid(bid, signal_I(bn), signal_E(bn), n_hyp+1, lambda_I_n, lambda_E_n, rho_I_n, rho_E_n);
                try
                    new_bid(:,bn) = fsolve(solve_hyp_bf_singlebid,single_bid,options_fsolve);
                catch
                    fprintf('Auction %d with %d bidders and bid grid point %d: Solving for bid failed! Resetting values...\n',t,n_hyp,bn);
                    if bn>1
                        new_bid(:,bn) = new_bid(:,bn-1);
                    elseif bn==1
                        new_bid(:,bn) = 0.0;
                    end
                end
            end

            % Split stacked vector of new bids into incumbent and entrant bids.
            bb_I = new_bid(1,:)';
            bb_E = new_bid(2,:)';
            b_min_I = max(0,max(bid_grid,bb_I));
            b_min_E = max(0,max(bid_grid,bb_E));
            fprintf('Calculating for gross auction %d and %d entrants entering...\n',t,n_hyp);
            
            % Compute Pr(efficient|winning) for incumbent.
            num_I = 1 - wblcdf(b_min_I,lambda_I_n,rho_I_n);
            num_I_2 = (1 - wblcdf(b_min_I,lambda_I_n,rho_I_n)) .* (1 - wblcdf(b_min_E,lambda_E_n,rho_E_n)).^(n_hyp-1);
            num_E = (1 - wblcdf(b_min_E,lambda_E_n,rho_E_n)).^(n_hyp);

            % Updated version of last term in EQ 60:
            num_IE = (1 - wblcdf(b_min_I,lambda_I_n,rho_I_n)) .* (1 - wblcdf(bid_grid,lambda_E_n,rho_E_n)).^(n_hyp-1);
            num_comp = [num_I, num_I_2,num_IE];

            %% SW: For diagnostics, plot pdfs of entrant and incumbent bid distribution over bid grid.
            plot(bid_grid,PDF_I,bid_grid,PDF_E);
            xline(bid_win(t),'-');
            title('Comparing bid distribution for incumbent and entrant');
            legend('Incumbent','Entrant','Observed winning bid');
            % Diagnose integral of bid distribution densities.
            int_pdf_e = trapz(bid_grid(1:end),PDF_E(1:end));
            int_pdf_i = trapz(bid_grid(1:end),PDF_I(1:end));

            % Compute vector of entrants' costs.
            signal_E = bid_grid - MU_E;
            plot(bid_grid,signal_E,bid_grid,signal_I);
            legend('Entrant','Incumbent');
            title('Evolution of signals over bid grid');
            xlabel('Bid');
            ylabel('Implied cost signal');
            
            % Put together terms as described in paper.
            RR = PDF_I ./ int_pdf_i .* num_E + n_hyp .* PDF_E ./int_pdf_e .* num_IE;
            RR = PDF_I .* num_E + n_hyp .* PDF_E .* num_IE;
            RR_I = PDF_I .* num_E;
            RR_E = n_hyp .* PDF_E .* num_IE;
            
            % Numerically integrate over the full bid grid.
            integral = trapz(bid_grid(1:end),RR(1:end));
            % Only for testing/debugging.
            integral_I = trapz(bid_grid,RR_I);
            integral_E = trapz(bid_grid,RR_E);


            % Diagnose integral of bid distribution densities.
            fprintf('Integral over predicted bid densities for incumbent and entrant is: %6.4f and %6.4f.\n', int_pdf_i,int_pdf_e);
            fprintf('Ex ante efficiency of auction %d with %d entrant(s) entering: %6.4f \n', t,n_hyp,integral);
            fprintf('Ex ante efficiency contribution of incumbent and entrant: %6.4f and %6.4f\n', integral_I,integral_E);

            % Write efficiency probability to container.
            Pr_eff_endo_n(t,n_hyp) = integral;
        end
    end % end loop over bidder configurations.
    %% Average over different bidder configurations.
    % Reweighting different N-configurations so that they sum up to one.
    prob_N_grid_rescaled = prob_N_grid(t,1:N_max_t) ./ sum(prob_N_grid(t,1:N_max_t));
    Pr_efficient_gross(t) = sum( Pr_eff_endo_n(t,1:N_max_t) .* prob_N_grid_rescaled);
    E_subsidy_gross(t) = sum( E_bid_endo_n(t,1:N_max_t) .* prob_N_grid_rescaled);
    % Extract relevant statistics for specification with constant number of bidders.
    Pr_efficient_gross_constant_n(t) = Pr_eff_endo_n(t,N(t)-1);
    sum_Pr_win_gross_constant_n(t) = sum_Pr_win_gross(t,N(t)-1);    
    E_subsidy_gross_constant_n(t) = E_bid_endo_n(t,N(t)-1);

    % If we want to allow for no entrant entering in auction:
    % If no entrant enters, then by definition efficiency is one.
    Pr_efficient_gross_alt(t) = (1-prob_no_entrant(t)) .* Pr_efficient_gross(t) + prob_no_entrant(t) * 1.0;
    % Winning bid is less clear, for now we simply extrapolate from our
    % estimated bid functions when setting N=1.
    E_subsidy_gross_alt(t) = (1-prob_no_entrant(t)) .* E_subsidy_gross(t) + prob_no_entrant(t) * E_bid_zeroN(t);
end % end loop over auctions.

% Comment out this line if we want to use the endogenous N case where we do
% not allow for no entrant entering.
Pr_efficient_gross_backup = Pr_efficient_gross;
Pr_efficient_gross = Pr_efficient_gross_alt;
E_subsidy_gross_backup = E_subsidy_gross;
E_subsidy_gross = E_subsidy_gross_alt;

%% Compare efficiency probabilities with constant and endogenous number of bidders.
% Column 1: with constant N
% Column 2: endogenous N, conditonal on at elast one entrant entering.
% Column 3: endogenous N, allowing that no entrant may enter.
Pr_efficient_gross_compare = [Pr_efficient_gross_constant_n, Pr_efficient_gross_backup, Pr_efficient_gross_alt];
E_bidwin_gross_compare = [E_subsidy_gross_constant_n, E_subsidy_gross, E_subsidy_gross_alt];

fprintf('Mean and median efficiency probabilities in gross auctions:\n %6.4f \t %6.4f \t for constant number of bidders\n%6.4f \t %6.4f \t for endogenous number of bidders \n%6.4f \t %6.4f \t for endogenous number of bidders conditional on at least one entrant entering\n',nanmean(Pr_efficient_gross_constant_n),nanmedian(Pr_efficient_gross_constant_n), nanmean(Pr_efficient_gross),nanmedian(Pr_efficient_gross), nanmean(Pr_efficient_gross_backup),nanmedian(Pr_efficient_gross_backup));
fprintf('Mean and median expected winning bid in gross auctions:\n %6.4f \t %6.4f \t for constant number of bidders\n%6.4f \t %6.4f \t for endogenous number of bidders\n%6.4f \t %6.4f \t for endogenous number of bidders conditional on at least one entrant entering',nanmean(E_subsidy_gross_constant_n),nanmedian(E_subsidy_gross_constant_n), nanmean(E_subsidy_gross),nanmedian(E_subsidy_gross), nanmean(E_subsidy_gross_backup),nanmedian(E_subsidy_gross_backup));

%% Compute expected ticket revenues from gross contracts.
% Load net-2-step parameters.
load(project_paths('OUT_ANALYSIS','theta_net2'));
% Construct revenue regressors.
% IMPORTANT: MAKE SURE THESE ARE THE SAME AS THE ONES USED IN ESTIMATION!
% X_revenue = [data(:,10)./10, log(data(:,8)*10) ./10, log(data(:,6))./10];
% X_revenue = [log(data(:,8)*10)];
X_revenue = [data(:,9)./10];

%% Construct table for expected revenues and variance.
% Mean and variance of parent normal distribution.
mean_rev_aux = ([ones(T,1), X_revenue] * theta_2_opt(end-size(X_revenue,2):end));
% Variance of revenue signal distribution.
sigma_rev_aux = sqrt(exp(theta_2_opt(end-2))).*ones(N_obs,1);
% Account for truncation to compute actual mean and variance of positive
% revenue distribution.
rsig_aux = mean_rev_aux ./ sigma_rev_aux;
denom = normcdf(rsig_aux);
mean_revenue = mean_rev_aux + (sigma_rev_aux .* normpdf(rsig_aux) ./ denom) ;
sigma_revenue = sqrt((sigma_rev_aux.^2 .* ( 1 - (rsig_aux .* normpdf(rsig_aux) ./ denom) - (normpdf(rsig_aux).^2 ./ denom.^2) )));
sigma_rev_gross = sigma_revenue;
% Save expected ticket revenues for gross contracts.
E_TR_gross = mean_revenue;

% Total zkm per contract, adjust units to be conformable with what is contained in estimation workspace.
total_zkm = data(:,8) .* 1E6 .* data(:,10)*10;
% Revenue per zkm.
rev_per_zkm = mean_revenue .* 1E7 ./ total_zkm;
fprintf('Gross auctions: Mean and median of revenue per zkm statistics: %6.4f and %6.4f.\n',mean(rev_per_zkm),median(rev_per_zkm));
fprintf('Gross auctions: Expected revenue (mean and median across gross auctions): %6.4f and %6.4f.\n',mean(mean_revenue),median(mean_revenue));
% Export revenue per ZKM
rev_per_zkm_gross = rev_per_zkm;
% Save in separate file.
save(project_paths('OUT_ANALYSIS','rev_est_gross'),'rev_per_zkm_gross','E_TR_gross','sigma_rev_gross');

% Compute expected agency payoff.
payoff_agency_gross = E_TR_gross - E_subsidy_gross;
payoff_agency_gross_constant_n = E_TR_gross - E_subsidy_gross_constant_n;
% Concatenate winning bid, expected subsidy, expected ticket revenue and
% agency payoff.
cf_revenue_gross = [bid_win, E_subsidy_gross, E_TR_gross, payoff_agency_gross];
% Old version with constant number of n.
cf_revenue_gross_constant_n = [bid_win, E_subsidy_gross_constant_n, E_TR_gross, payoff_agency_gross_constant_n];
mean(cf_revenue_gross)
median(cf_revenue_gross)
mean(cf_revenue_gross_constant_n)
median(cf_revenue_gross_constant_n)
% Safely store number of auctions in gross auction sample.
N_gross = N;
% Save workspace.
save(project_paths('OUT_ANALYSIS','cf_prep_gross_auctions'));