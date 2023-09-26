%{
    Counterfactual preparation for net auction sample.
    Compute probability of choosing efficient firm in gross auctions and expected subsidies and agency payoffs.
    
%}

clear
clc
format('short');
% Define necessary globals.
global N_obs K
% Incumbent in first column, entrant in second column.
clf('reset')
% Set seed in case any simulation is used.
rng(123456);

%% Load gross auction workspace and net auction parameters.
load(project_paths('OUT_ANALYSIS','net_step1'));
load(project_paths('OUT_ANALYSIS','postestimation_workspace_net.mat'));
% Number of auctions in sample.
T = length(db_win);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load probabilities of different bidder configurations in net auctions.
load(project_paths('OUT_ANALYSIS','na_entry_npot'));
N_pot_max = max(N_pot_net);

load(project_paths('OUT_ANALYSIS','na_entry_n_probs'));
prob_N_grid = prob_N_grid_net(:,1:N_pot_max);
check_prob_net_N = sum(prob_N_grid,2);
prob_no_entrant = 1 - check_prob_net_N;

% Analyze how number of bidders is likely to change.
% Generally, it's not clear that number of bidders would go
% up or down.
comp_N = [N-1, prob_N_grid];

%% Initialize containers for different efficiency probabilities and expected subsidies.
size_bidgrid = 400;
% Container for final probability of interest.
Pr_efficient_net = zeros(T,1);
% Alternative version in which we allow for no entrant entering as possibility.
Pr_efficient_net_alt = zeros(T,1);
Pr_efficient_net_constant_n = zeros(T,1);
sum_Pr_win_net = zeros(T,N_pot_max);
sum_Pr_win_net_constant_n = zeros(T,1);
% Expected subsidies.
E_subsidy_net = zeros(T,1);
E_subsidy_net_constant_n = zeros(T,1);
% Alternative version in which we allow for no entrant entering as possibility.
E_subsidy_net_alt = zeros(T,1);
% Container for expected bid for each auction and potential bidder configuration.
E_bid_endo_n = zeros(T,N_pot_max);
% Container for efficiency probability for each auction and potential bidder configuration.
Pr_eff_endo_n = zeros(T,N_pot_max);
% Aux object for debugging.
test_Pr_eff_dens_endo_n = zeros(T,N_pot_max);
%% Create containers for case when no entrant enters.
% By construction efficiency probability is one. 
% For revenue, we can compute the expected bid based on the bid distribution extrapolated to N=1.
E_bid_zeroN = zeros(T,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Replicate bid function computation in order to compute rho_inv(b_grid).

% Compute bid function parameters for all lines.
[~, lambda_vec_sim_I, lambda_vec_sim_E, rho_vec_sim_I, rho_vec_sim_E] = sim_wb_param(X_orig,theta_opt);

% Containers for winning bid signals.
g_I_bf = zeros(N_obs,size_bidgrid);
G_I_bf = zeros(N_obs,size_bidgrid);
g_E_bf = zeros(N_obs,size_bidgrid);
G_E_bf = zeros(N_obs,size_bidgrid);
% Container for RHO-signal including the X_IE-term.
signal_I_bf = zeros(N_obs,size_bidgrid);
signal_E_bf = zeros(N_obs,size_bidgrid);
% Container for own signal: rho (excluding the X_IE-term).
signal_I_rho = zeros(N_obs,size_bidgrid);
signal_E_rho = zeros(N_obs,size_bidgrid);

% Compute revenue mean, variane and alpha parameters to put into X_IE
% routine.
% These should be mean and variance of the parent normal distribution, not
% the truncated one!
R_nbf = mean_rev_aux;
% We need standard deviation of parent normal of revenue distribution, not variance and not the SD of the truncated revenue distribution!
sigma_r_bnf = sigma_rev_aux;
% Containers for terms in final computation.
E_term_n = zeros(N_obs,size_bidgrid,2,N_pot_max);
% Use to extract E-terms for single bidder types here.
E_term_n_I = zeros(N_obs,size_bidgrid,N_pot_max);
E_term_n_E = zeros(N_obs,size_bidgrid,N_pot_max);
% Initialize container for components of single bidder.
X_IE_bnf = zeros(N_obs,4,size_bidgrid);
X_IE_bnf_test = zeros(N_obs,4,size_bidgrid);
% Set options for solving for X_IE.
% Restrict solution to be nonnegative.
lb = [-1;-1;-1;-1];
% Set options for solver.
options=optimoptions('lsqnonlin','Display','off');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize empty bid grid for each line.
bid_grid_full = zeros(T, size_bidgrid);


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
    fprintf('Computing efficiency for auction %d...\n',t);
    % This is the maximum TOTAL number of bidders that we could observe in
    % the auction based on our data.
    N_max_t = N_pot_net(t);
    % Prepare objects for computation of bid function parameters.
    X_aux_endo_n = X_aux(t,:);

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
    bg_min = min(bid_min_I,bid_min_E);
    bg_max = max(bid_max_I,bid_max_E);
    bid_grid=linspace(bg_min,bg_max,size_bidgrid)';
    % Compute various CDFs and PDFs.
    CDF_E = wblcdf(bid_grid,lambda_E_n,rho_E_n);
    PDF_E = wblpdf(bid_grid,lambda_E_n, rho_E_n);
    CDF_I = wblcdf(bid_grid,lambda_I_n,rho_I_n);
    PDF_I = wblpdf(bid_grid,lambda_I_n, rho_I_n);
    E_bid_zeroN(t,1)  = trapz(bid_grid,bid_grid .* PDF_I);
    % end computation of bid when no entrant enters.

    %% LOOP OVER ALL POTENTIAL N when at least one bidder enters.
    % n_hyp indicates the total number of entrants entering the auction.
    % N_max_t is the maximum number of entrants potentially entering.
    for n_hyp=1:N_max_t
        % Define correct alpha parameters.
        % First element for incumbent, second element for entrant.
        alpha_bnf_n = [alpha_I_grid(min(n_hyp,4)),alpha_E_grid(min(n_hyp,4))];
        % Recall that in bid function, parameters use min(n,5).
        X_aux_endo_n(end) = log(min(n_hyp+1,5)) ./ 10;
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
        bg_min = min(bid_min_I,bid_min_E);
        bg_max = max(bid_max_I,bid_max_E);
        bid_grid=linspace(bg_min,bg_max,size_bidgrid)';

        bid_grid_full(t,:) = linspace(bg_min,bg_max,size_bidgrid)';
        bid_grid = bid_grid_full(t,:)';

        % Test whether one bidder type dominates the other.
        dominant_entrant = (bid_max_E < bid_min_I);
        dominant_incumbent = (bid_max_I < bid_min_E);
        auction_indicator(t,n_hyp) = (auction_indicator(t,n_hyp) - 44 .* dominant_entrant);
        auction_indicator(t,n_hyp) = (auction_indicator(t,n_hyp) - 88 .* dominant_incumbent);
        % Extract relevant bid function parameters for auction t.
        lambda_I_sim = lambda_I_n;  
        rho_I_sim = rho_I_n;
        lambda_E_sim = lambda_E_n;
        rho_E_sim = rho_E_n;
        
        %% Compute markups for each bid.
        [CDF_I, PDF_I] = eval_bf(lambda_I_sim,rho_I_sim,bid_grid,truncation_bid);
        [CDF_E, PDF_E] = eval_bf(lambda_E_sim,rho_E_sim,bid_grid,truncation_bid);
        % Compute probabilities of incumbent and entrant winning for bid_grid.
        P_I_win = PDF_I .* (1-CDF_E).^(n_hyp);
        P_E_win = (n_hyp) .* PDF_E .* (1-CDF_E).^(n_hyp-1) .* (1-CDF_I);
        % Check that winning probabilities add up to one.
        sum_Pr_win_net(t,n_hyp) = trapz(bid_grid,P_E_win) + trapz(bid_grid,P_I_win);
                 
        % Numerator of incumbent's markup term.
        G_MI_bf = (1-CDF_E).^(n_hyp);
        % Denominator of entrant's markup term.
        g_MI_bf = (n_hyp) .* (1-CDF_E).^(n_hyp-1) .* PDF_E;
        % Write numerator and denominator into container.
        g_I_bf(t,:) = g_MI_bf;
        G_I_bf(t,:) = G_MI_bf;
        % Compute incumbent's markup.
        MU_I_bf = G_MI_bf ./ g_MI_bf;
        % Compute incumbent's winning cost signal.
        signal_I_bf_aux = bid_grid - MU_I_bf;
        % Set markups when losing with Pr 1 to zero.
        signal_I_bf_aux(G_I_bf(t,:)==0) = bid_grid(G_I_bf(t,:)==0);
        signal_I_bf(t,:) = signal_I_bf_aux;
        % Compute markup term for entrants.
        % Numerator of entrant's markup term.
        G_ME_bf = (1-CDF_E).^(n_hyp-1) .* (1-CDF_I);
        % Denominator of entrant's markup term.
        if n_hyp==1
            g_ME_bf = PDF_I .* (1-CDF_E).^(n_hyp-1);
        else
            g_ME_bf = (n_hyp-1) .* (1-CDF_E).^(n_hyp-2) .* PDF_E  .* (1-CDF_I) + PDF_I .* (1-CDF_E) .^ (n_hyp-1);
        end
        % Write numerator and denominator into container.
        G_E_bf(t,:) = G_ME_bf;
        g_E_bf(t,:) = g_ME_bf;
        % Compute markup for entrant.
        MU_E_bf = G_ME_bf ./ g_ME_bf;
        % Safety measure to avoid weird markup terms.
        MU_E_bf(g_ME_bf==0 & G_ME_bf==0) = 0;
        MU_I_bf(g_MI_bf==0 & G_MI_bf==0) = 0;
        % Compute vector of entrants' costs.
        signal_E_bf_aux = bid_grid - MU_E_bf;
        % Set markups for bidds that surely lose to zero.
        signal_E_bf_aux(G_E_bf(t,:)==0) = bid_grid(G_E_bf(t,:)==0);
        signal_E_bf(t,:) = signal_E_bf_aux;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Compute expected subsidy paid by the agency.
        E_bid_endo_n(t,n_hyp) = trapz(bid_grid,bid_grid .* (PDF_I .* G_MI_bf + n_hyp .* PDF_E .* G_ME_bf));
        
        % Prep work for computing efficiency probabilities.    
        for b=1:size(bid_grid)
            % Adjust signals for X_IE to get rho (net of expectation term).
            % Compute X_IE for each grid point.
            X_IE_start = [0.9.*R_nbf(t);0.8 .* R_nbf(t);0.45.*R_nbf(t);0.4 .* R_nbf(t);];
            RHO_signal = [signal_I_bf(t,b); signal_E_bf(t,b)];
            % Anonymous function to input data and auxiliary parameters.
            solve_E_FP_a = @(X_IE) solve_E_FP(X_IE, alpha_bnf_n(2), alpha_bnf_n(1), R_nbf(t), sigma_r_bnf(t), RHO_signal, kdens_container(:,t,:,:), n_hyp+1,db_prob(t));
            % Compute fixed point of system describing beliefs about rivals'
            % revenue signals: use lsqnonlin instead of fsolve to impose
            % nonnegativity constraints/
            % CRBF: replaced this by not writing to _test object before!
            X_IE_bnf(t,:,b) = lsqnonlin(solve_E_FP_a,X_IE_start,lb,[],options);
            % New order of terms in fixed point search:
            % 1. X_I_eq
            % 2. X_E_eq
            % 3. X_I_plus
            % 4. X_E_plus
            
            % Construct sums for decomposing compound signal RHO.
            % For incumbent.
            E_term_n(t,b,1,n_hyp) = (n_hyp-1) .* alpha_bnf_n(2) .* X_IE_bnf(t,4,b) + alpha_bnf_n(2) .* X_IE_bnf(t,2,b);
            % For entrant.
            if n_hyp==1 % distinguish special case for E-term when only 2 bidders.
                E_term_n(t,b,2,n_hyp) = alpha_bnf_n(1) .* X_IE_bnf(t,1,b);
            elseif n_hyp>1
                E_term_n(t,b,2,n_hyp) = db_prob(t) .*  (alpha_bnf_n(1) .* X_IE_bnf(t,1,b) + (n_hyp-1) .* alpha_bnf_n(2) .* X_IE_bnf(t,4,b)) + ...
                (1-db_prob(t)) .* (alpha_bnf_n(1) .* X_IE_bnf(t,3,b) + alpha_bnf_n(2) .* X_IE_bnf(t,2,b) + (n_hyp-2) .* alpha_bnf_n(2) .* X_IE_bnf(t,4,b));  
            end
        end % end loop over bid grid
        
        
        % Copy E-terms to separate vectors for entrant and incumbent for
        % investigation.
        E_term_n_I(t,:,n_hyp) = E_term_n(t,:,1,n_hyp);
        E_term_n_E(t,:,n_hyp) = E_term_n(t,:,2,n_hyp);
        
        % Adjust RHO for X_IE values.
        signal_I_rho(t,:) = signal_I_bf(t,:) + E_term_n(t,:,1,n_hyp);
        signal_E_rho(t,:) = signal_E_bf(t,:) + E_term_n(t,:,2,n_hyp);
        
        % Safety measures to avoid too negative (net) cost signals.
        % CRBF: This could matter...print if we hit this!
        net_cost_signal_threshold = -5;
        sum_trunc_I = sum(signal_I_rho(t,:)<net_cost_signal_threshold);
        sum_trunc_E = sum(signal_E_rho(t,:)<net_cost_signal_threshold);
        if sum_trunc_I>0 || sum_trunc_E>0
            fprintf('CAREFUL: NET COST SIGNAL TRUNCATED FOR AUCTION %d! INVESTIGATE!\n',t);
%             fprintf('Number of truncated net cost signal for incumbent ent entrant, respectively: %6.4f \t %6.4f.\n',sum_trunc_I,sum_trunc_E);
        end
        signal_I_rho(t,signal_I_rho(t,:)<net_cost_signal_threshold) = net_cost_signal_threshold;
        signal_E_rho(t,signal_E_rho(t,:)<net_cost_signal_threshold) = net_cost_signal_threshold;
        
        % This is not needed for computation of efficiency probabilities.
        % But might be useful for debugging.
        bid_function_E = [signal_E_rho(t,:)' bid_grid];
        bid_function_I = [signal_I_rho(t,:)' bid_grid];
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Compute probability of winner being cost-efficient firm.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % This is an integral over all potential winning bids, winner's costs,
        % rivals' compound signals, and rivals' costs.
        if dominant_entrant==1 % entrant always wins the auction.
            fprintf('Calculating for net auction %d and %d entrants entering...\n',t,n_hyp);
            fprintf('Entrant has dominant bid distribution, efficiency prob purely based on cost distributions!\n');
            % Entrant always wins, so efficiency probability is probability
            % the the lowest cost entrant has lower cost than the incumbent
            c_grid = ksdens_I_grid(t,:);
            pdf_c_i = ksdens_I(t,:);
            cdf_c_e = ksCDF_E(t,:);
            % Compute probability that incumbent has a cost lower than the
            % lowest cost entrant. This is the probability of having an
            % inefficient auction.
            RR_ineff_E = pdf_c_i .* (1.0-cdf_c_e).^n_hyp;
            Pr_e_ineff = trapz(c_grid,RR_ineff_E);
            % Write efficiency probability to container.
            Pr_eff_endo_n(t,n_hyp) = 1.0 - Pr_e_ineff;
            fprintf('Pr that winning entrant is efficient (based on cost distribution): %6.4f.\n',Pr_eff_endo_n(t,n_hyp));
        elseif dominant_incumbent==1 % incumbent always wins the auction.
            fprintf('Calculating for net auction %d and %d entrants entering...\n',t,n_hyp);
            fprintf('Incumbent has dominant bid distribution, efficiency prob purely based on cost distributions!\n');
            % Incumbent always wins, so efficiency probability is probability
            % the the lowest cost entrant has higher cost than the
            % incumbent.
            c_grid = ksdens_I_grid(t,:);
            pdf_c_i = ksdens_I(t,:);
            cdf_c_e = ksCDF_E(t,:);
            % Compute probability that incumbent has a cost lower than the
            % lowest cost entrant. This is the probability of having an
            % efficient auction.
            RR_eff_I = pdf_c_i .* (1.0-cdf_c_e).^n_hyp;
            Pr_i_eff = trapz(c_grid,RR_eff_I);
            % Write efficiency probability to container.
            Pr_eff_endo_n(t,n_hyp) = Pr_i_eff;
            fprintf('Pr that winning incumbent is efficient (based on cost distribution): %6.4f.\n',Pr_eff_endo_n(t,n_hyp));
        else % "normal" case where both bidder types could win.
            % Start with some helper objects describing distribution of costs and
            % revenue signals for all bidder types.
            %%%%%%%%%%%%%%%%
            % For incumbent.
            %%%%%%%%%%%%%%%%
            % Extract full cost distribution.
            f_c_supp_I = kdens_container(:,t,1,2);
            f_c_I = kdens_container(:,t,1,1);
            test_int_f_c_I = trapz(f_c_supp_I,f_c_I);
            % Compute revenue signal implied by specific c-rho combination.
            % Note that rho already excludes the E-terms.
            f_r_arg_I = (repmat(f_c_supp_I,1,size_bidgrid) - repmat(signal_I_rho(t,:),kd_no_gridpoints,1)) ./ alpha_bnf_n(1); 
            
            % Compute revenue signal density at implied r-values.
            % Use mean and standard deviation of parent normal distribution here.
            f_r_aux_I = normpdf((f_r_arg_I-R_nbf(t))./sigma_r_bnf(t)) ./ sigma_r_bnf(t) ./ normcdf(R_nbf(t)./sigma_r_bnf(t));
            % Combine zero and non-zero parts of density of revenue signal.          
            f_r_I = f_r_aux_I .* (f_r_arg_I >= 0) + 0 .* (f_r_arg_I < 0);
            % Adjust cost density to condition only on possible rho-c
            % combinations with positive r value.
            cd_cond_aux_I = repmat(f_c_I,1,size_bidgrid);
            neg_r_index_I = (f_r_arg_I < 0);
            mass_ex_cost_I = sum(neg_r_index_I .* cd_cond_aux_I) ./ sum(cd_cond_aux_I);
            % Fix if all revenue signals are negative.
            mass_ex_cost_I(mass_ex_cost_I==1) = 0;
            cd_cond_aux_I = (1-neg_r_index_I) .* cd_cond_aux_I;
            cdens_cond_I = cd_cond_aux_I ./ repmat((1-mass_ex_cost_I),kd_no_gridpoints,1);
            % Combine cost and revenue density and integrate
            cr_density_I = cdens_cond_I .* f_r_I;
            %cr_density_I = f_c_I .* f_r_I; 
            test_cr_density_I = trapz(f_c_supp_I,cr_density_I);
            % Quick fix for nonsensical combinations.
            cr_density_I(isnan(cr_density_I)) = 0;

            %%%%%%%%%%%%%%
            % For entrant.
            %%%%%%%%%%%%%%
            f_c_supp_E = kdens_container(:,t,2,2);
            f_c_E = kdens_container(:,t,2,1);
            test_int_f_c_E = trapz(f_c_supp_E,f_c_E);
            % Compute revenue signal implied by specific c-rho combination.
            % Note that rho already excludes the E-terms.
            f_r_arg_E = (repmat(f_c_supp_E,1,size_bidgrid) - repmat(signal_E_rho(t,:),kd_no_gridpoints,1)) ./ alpha_bnf_n(2); 
            % Compute revenue signal density at implied r-values.
            f_r_aux_E = normpdf((f_r_arg_E-R_nbf(t))./sigma_r_bnf(t)) ./ sigma_r_bnf(t) ./ normcdf(R_nbf(t)./sigma_r_bnf(t));
            % Combine zero and non-zero parts of density of revenue signal.          
            f_r_E = f_r_aux_E .* (f_r_arg_E >= 0) + 0 .* (f_r_arg_E < 0);
            % Adjust cost density to condition only on possible rho-c
            % combinations with positive r value.
            cd_cond_aux_E = repmat(f_c_E,1,size_bidgrid);
            neg_r_index = (f_r_arg_E < 0);
            mass_ex_cost_E = sum(neg_r_index .* cd_cond_aux_E) ./ sum(cd_cond_aux_E);
            % Fix if all revenue signals are negative.
            mass_ex_cost_E(mass_ex_cost_E==1) = 0;
            cd_cond_aux_E = (1-neg_r_index) .* cd_cond_aux_E;
            cdens_cond_E = cd_cond_aux_E ./ repmat((1-mass_ex_cost_E),kd_no_gridpoints,1);
            % Combine cost and revenue density and integrate
            cr_density_E = cdens_cond_E .* f_r_E; 
            test_cr_density_E = trapz(f_c_supp_E,cr_density_E);
            % Quick fix for nonsensical combinations. This does not affect counterfactual simulation.
            cr_density_E(isnan(cr_density_E)) = 0;
            % Scale up cr-density to that it integrates to one.
            cr_density_I = cr_density_I ./ repmat(test_cr_density_I,kd_no_gridpoints,1);
            cr_density_E = cr_density_E ./ repmat(test_cr_density_E,kd_no_gridpoints,1);

            %% Compute innermost integral: over losers' cost.
            % Compute probability that all losers costs are larger than winner's cost.
            % For a given combination of rho (of losers) and c (of winner).
            % Initialize containers.
            int_C_I = zeros(kd_no_gridpoints,size_bidgrid);
            int_C_E = int_C_I;

            % Integrate over c-r densities.
            % Loop over all winner's cost.
            for i=1:kd_no_gridpoints
                % Pick all losers' costs that are larger than winner's cost.
                % For entrant.
                cl_index_I = (f_c_supp_E > f_c_supp_I(i));
                cr_density_E_trim = cr_density_E .* repmat(cl_index_I,1,size_bidgrid);
                int_C_E(i,:) = trapz(f_c_supp_E, cr_density_E_trim);
                % For incumbent.
                cl_index_E = (f_c_supp_I > f_c_supp_E(i));
                cr_density_I_trim = cr_density_I .* repmat(cl_index_E,1,size_bidgrid);
                int_C_I(i,:) = trapz(f_c_supp_I, cr_density_I_trim);
            end

            %% Compute second-innermost integral (B integrals):
            % Probability that all rivals have higher net cost signals rho, i.e.
            % they lose,
            % given the winner's net cost signal rho and the winner's cost. 
            % Integrate over all rivals' rho for a given rho-cost realization of winner. 
            % Initialize container for B-integrals.
            int_B_I = zeros(kd_no_gridpoints,size_bidgrid);
            int_B_E = zeros(kd_no_gridpoints,size_bidgrid);    

            % Compute integral over rho density.
            % Compute density of revenue part of signal.        
            % Own revenue signal as function of c, RHO and E-terms.
            f_r_rho_arg_I = (repmat(f_c_I,1,size_bidgrid) - repmat(signal_I_rho(t,:),kd_no_gridpoints,1) ) ./ alpha_bnf_n(1);
            % Evaluate density of revenue signal based on truncated normal.
            f_r_rho_aux_I = normpdf((f_r_rho_arg_I-R_nbf(t))./sigma_r_bnf(t)) ./ sigma_r_bnf(t) ./ normcdf(R_nbf(t)./sigma_r_bnf(t));
            % Combine zero and non-zero parts of density of revenue signal.          
            f_r_rho_I = f_r_rho_aux_I .* (f_r_rho_arg_I >= 0) + 0 .* (f_r_rho_arg_I < 0);
            f_c_alpha_r_I = cdens_cond_I .* f_r_rho_I;
            f_rho_I = trapz( f_c_supp_I, f_c_alpha_r_I)'; 
            % Same for entrants.
            % Own revenue signal as function of c, RHO and E-terms.
            f_r_rho_arg_E = (repmat(f_c_E,1,size_bidgrid) - repmat(signal_E_rho(t,:),kd_no_gridpoints,1) ) ./ alpha_bnf_n(2);
            % Evaluate density of revenue signal based on truncated normal.
            f_r_rho_aux_E = normpdf((f_r_rho_arg_E-R_nbf(t))./sigma_r_bnf(t)) ./ sigma_r_bnf(t) ./ normcdf(R_nbf(t)./sigma_r_bnf(t));

            % Combine zero and non-zero parts of density of revenue signal.          
            f_r_rho_E = f_r_rho_aux_E  .* (f_r_rho_arg_E >= 0) + 0 .* (f_r_rho_arg_E < 0);
            f_c_alpha_r_E = cdens_cond_E .* f_r_rho_E;
            f_rho_E = trapz( f_c_supp_E, f_c_alpha_r_E)'; 
            % Testing/debugging: test whether rho_I and rho_E densities integrate to
            % one.
            test_f_rho_I_int = trapz(signal_I_rho(t,:)',f_rho_I');
            test_f_rho_E_int = trapz(signal_E_rho(t,:)',f_rho_E');

            % Condition to integrate f_rho distribution densties to one.
            f_rho_I = f_rho_I ./ test_f_rho_I_int;
            f_rho_E = f_rho_E ./ test_f_rho_E_int;

            % Loop over all winner's cost realizations and potential rho-signnals
            % of loser.
            % Careful: need to integrate over other type's rho distribution here.
            % Make copy of density vector that can safely be set to zero
            % progressively.
            f_rho_I_loop = f_rho_I;
            f_rho_E_loop = f_rho_E;
            for i_rho=1:size_bidgrid
                f_rho_I_loop(1:i_rho,1) = 0;
                % Compute conditioning integral.
                f_rho_I_int_loop = trapz(signal_I_rho(t,:)',f_rho_I_loop);
                % Fix for pathetic cases.
                f_rho_I_int_loop(f_rho_I_int_loop==0) = 1;
                % For entrant.
                f_rho_E_loop(1:i_rho,1) = 0;
                % Compute conditioning integral.
                f_rho_E_int_loop = trapz(signal_E_rho(t,:)',f_rho_E_loop);
                % Fix for pathetic cases.
                f_rho_E_int_loop(f_rho_E_int_loop==0) = 1;
                for i_cost=1:kd_no_gridpoints
                    % Pick appropriate C-integral for incumbent.
                    C_int_I_aux = int_C_I(i_cost,i_rho)';
                    integrand_B_I = f_rho_I_loop .* C_int_I_aux ./ f_rho_I_int_loop;
                    int_B_I(i_cost,i_rho) = trapz(signal_I_rho(t,:)', integrand_B_I);
                    % Pick appropriate C-integral for entrant.
                    C_int_E_aux = int_C_E(i_cost,i_rho)';
                    integrand_B_E = f_rho_E_loop .* C_int_E_aux ./ f_rho_E_int_loop;
                    int_B_E(i_cost,i_rho) = trapz(signal_E_rho(t,:)', integrand_B_E);
                end
            end

            %% Compute second-outermost integral (A integrals): over all potential winner's costs.
            % Check whether density in A-integrals integrates to one. 
            test_density_A_I = trapz(f_c_supp_I, cr_density_I);% ./ repmat(cr_int_A_I,kd_no_gridpoints,1));
            test_density_A_E = trapz(f_c_supp_E, cr_density_E); % ./ repmat(cr_int_A_E,kd_no_gridpoints,1));

            % Fix NaNs in cr_density.
            cr_density_I(isnan(cr_density_I)) = 0;
            cr_density_E(isnan(cr_density_E)) = 0;
            % Fix NaNs in int_B.
            int_B_I(isnan(int_B_I))=0;
            int_B_E(isnan(int_B_E))=0;

            int_A_I = trapz(f_c_supp_I, int_B_E.^(n_hyp) .* cr_density_I)';
            int_A_E = trapz(f_c_supp_E, int_B_E.^(n_hyp-1) .* int_B_I .* cr_density_E)'; 

            
            %% Final integrand: integrate over all potential winning bids (or alternatively: rhos) and both bidder types.
            integrand_final = int_A_I .* PDF_I .* G_MI_bf + int_A_E .* (n_hyp) .* PDF_E .* G_ME_bf;
            test_integrand_final = PDF_I .* G_MI_bf + (n_hyp) .* PDF_E .* G_ME_bf;
            % Integrate over the full bid grid and write probability of selecting efficient bidder to vector.
            Pr_eff_endo_n(t,n_hyp) = trapz(bid_grid,integrand_final);
            test_Pr_eff_dens_endo_n(t,n_hyp) = trapz(bid_grid,test_integrand_final);
        end
    end % end loop over bidder configurations.



    % Average over different bidder configurations.
    % Reweighting different N-configurations so that they sum up to one.
    prob_N_grid_rescaled = prob_N_grid(t,1:N_max_t) ./ sum(prob_N_grid(t,1:N_max_t));
    Pr_efficient_net(t) = sum( Pr_eff_endo_n(t,1:N_max_t) .* prob_N_grid_rescaled);
    E_subsidy_net(t) = sum( E_bid_endo_n(t,1:N_max_t) .* prob_N_grid_rescaled);
    % Extract relevant statistics for specification with constant number of bidders.
    Pr_efficient_net_constant_n(t) = Pr_eff_endo_n(t,N(t)-1);
    sum_Pr_win_net_constant_n(t) = sum_Pr_win_net(N(t)-1);
    E_subsidy_net_constant_n(t) = E_bid_endo_n(t,N(t)-1);
    
    fprintf('Ex-ante efficiency of auction %d: %6.4f.\n',t, Pr_efficient_net(t));
    fprintf('Expected winning bid for auction %d: %6.4f.\n',t, E_subsidy_net(t));

    % If we want to allow for no entrant entering in auction:
    % If no entrant enters, then by definition efficiency is one.
    Pr_efficient_net_alt(t) = (1-prob_no_entrant(t)) .* Pr_efficient_net(t) + prob_no_entrant(t) * 1.0;
    % Winning bid is less clear, for now we simply extrapolate from our
    % estimated bid functions when setting N=1.
    E_subsidy_net_alt(t) = (1-prob_no_entrant(t)) .* E_subsidy_net(t) + prob_no_entrant(t) * E_bid_zeroN(t);
end % end loop over auctions.
Pr_efficient_net_backup = Pr_efficient_net;
Pr_efficient_net = Pr_efficient_net_alt;
E_subsidy_net_backup = E_subsidy_net;
E_subsidy_net = E_subsidy_net_alt;

%% Print output of counterfactual simulations.
% Compute some summary statistics of net efficiency probabilities.
disp('---STATISTICS FOR CF WITH ENDOGENOUS N---');
fprintf('Number of efficiency probabilities smaller than 2 percent: %d \n ', sum(Pr_efficient_net<0.02))
fprintf('Number of efficiciency probabilities above 10 percent: %d \n ', sum(Pr_efficient_net>0.1))
fprintf('Number of efficiciency probabilities larger than random assignment: %d \n ', sum(Pr_efficient_net>1./N))
fprintf('Mean efficiency probability (raw): %d \n ', mean(Pr_efficient_net))

disp('---STATISTICS FOR CF WITH CONSTANT N---');
fprintf('Number of efficiency probabilities smaller than 2 percent: %d \n ', sum(Pr_efficient_net_constant_n<0.02))
fprintf('Number of efficiciency probabilities above 10 percent: %d \n ', sum(Pr_efficient_net_constant_n>0.1))
fprintf('Number of efficiciency probabilities larger than random assignment: %d \n ', sum(Pr_efficient_net_constant_n>1./N))
fprintf('Mean efficiency probability (raw): %d \n ', mean(Pr_efficient_net_constant_n))

% Compare efficiency probabilities of net auctions.
% Column 1: constant N
% Column 2: endogenous N conditional on at least one entrant entering.
% Column 3: endogenous N allowing for possibility of no entrant entering.
Pr_efficient_net_compare = [Pr_efficient_net_constant_n, Pr_efficient_net, Pr_efficient_net_backup];
E_bidwin_net_compare = [E_subsidy_net_constant_n, E_subsidy_net, E_subsidy_net_backup];

fprintf('Mean and median efficiency probabilities in net auctions:\n %6.4f \t %6.4f \t for constant number of bidders\n%6.4f \t %6.4f \t for endogenous number of bidders \n%6.4f \t %6.4f \t for endogenous number of bidders conditional on at least one entrant entering\n',nanmean(Pr_efficient_net_constant_n),nanmedian(Pr_efficient_net_constant_n), nanmean(Pr_efficient_net),nanmedian(Pr_efficient_net_backup), nanmean(Pr_efficient_net_alt),nanmedian(Pr_efficient_net_backup));
fprintf('Mean and median expected winning bid in net auctions:\n %6.4f \t %6.4f \t for constant number of bidders\n%6.4f \t %6.4f \t for endogenous number of bidders\n%6.4f \t %6.4f \t for endogenous number of bidders conditional on at least one entrant entering',nanmean(E_subsidy_net_constant_n),nanmedian(E_subsidy_net_constant_n), nanmean(E_subsidy_net),nanmedian(E_subsidy_net_backup), nanmean(E_subsidy_net_alt),nanmedian(E_subsidy_net_backup));

% Drop outlier observations, if there are any.
Pr_efficient_net_clean = Pr_efficient_net;
Pr_efficient_net_clean(Pr_efficient_net_clean<0.02) = [];
fprintf('Mean efficiency probability (cleaned) with endogenous N: %d \n', mean(Pr_efficient_net_clean))

Pr_efficient_net_clean_constant_n = Pr_efficient_net;
Pr_efficient_net_clean_constant_n(Pr_efficient_net_clean_constant_n<0.02) = [];
fprintf('Mean efficiency probability (cleaned) with constant N: %d \n', mean(Pr_efficient_net_clean_constant_n))

% Plot distribution of efficiency probabilities.
subplot(2,1,1)
hist(Pr_efficient_net_clean, 12)
title('Histogram of efficiency probabilities (net auctions, clean, endogenous N)');
% Plot distribution of efficiency probabilities.
subplot(2,1,2)
hist(Pr_efficient_net_clean_constant_n, 12)
title('Histogram of efficiency probabilities (net auctions, clean, constant N)');

%% Extract expected ticket revenues.
E_TR_net = mean_revenue;
% Compute expected agency payoff.
payoff_agency_net = - E_subsidy_net;
payoff_agency_net_constant_n = - E_subsidy_net_constant_n;
% Concatenate winning bid, expected subsidy, expected ticket revenue and
% agency payoff.
cf_revenue_net = [bid_win, E_subsidy_net, E_TR_net, payoff_agency_net];
% Old version with constant number of n.
cf_revenue_net_constant_n = [bid_win, E_subsidy_net_constant_n, E_TR_net, payoff_agency_net_constant_n];
% Safely store number of observations in net auction sample.
N_net = N;
% Save workspace.
save(project_paths('OUT_ANALYSIS','cf_prep_net_auctions'));


