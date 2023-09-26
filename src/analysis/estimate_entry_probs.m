%{
    Estimate entry probabilities, i.e., get probability distribution over bidder configurations for joint sample (gross and net).

%}

clear
clc
format('short');

% Define necessary globals.
global data N_obs K

% Set seed in case any simulation is used.
rng(123456);
% SET PAPER SIZE FOR FIGURES TO AVOID LARGE MARGINS.
figuresize(15,12,'cm')
% Set options for MATLAB optimizers.
fminsearch_options = optimset('Display','iter-detailed', ...
                   'TolFun', 1E-4,'TolX',1E-8,'MaxFunEvals', 17500, 'MaxIter', 15000);
fminunc_options = optimset('Display','iter-detailed','TolFun', 1E-6,'TolX',1E-6, 'MaxFunEvals', 12000, 'MaxIter', 1500);

% Load data for gross auctions.
load(project_paths('OUT_ANALYSIS','grossauction_workspace'));
% Prepare data for gross auction sample.
N_pot_gross = N_pot_matrix(:,end-1);
N_gross = N; % number of observed bidders.
% Extract maximum number of potential entrants across auctions.
N_pot_max_gross = max(N_pot_gross);
% Version with constant and dummy for net auctions
X_entry_gross = [ones(size(data,1),1), zeros(size(data,1),1), log(data(:,10)), data(:,7), data(:,8), data(:,13), log(data(:,6))./10, N_pot_gross];
% Load data for net auctions.
load(project_paths('OUT_ANALYSIS','net_step1'));
%% Prepare relevant data for gross auction sample.
N_pot_net = N_pot_matrix(:,end-1);
N_net = N; % number of observed bidders in net auctions.
% Extract maximum number of potential entrants across auctions.
N_pot_max_net = max(N_pot_net);
% Add two dummies: one for constant term and one for net auction group.
X_entry_net = [ones(size(data,1),2), log(data(:,10)), data(:,7), data(:,8), data(:,13), log(data(:,6))./10, N_pot_net];

%% Join data from the two samples.
X_entry_joint = [X_entry_gross;X_entry_net];
N_pot_joint = [N_pot_gross;N_pot_net];
N_pot_max_joint = max(N_pot_max_gross,N_pot_max_net);
N_joint = [N_gross;N_net];

%% Prep of joint estimation of entry probabilities.
% Number of total parameters to estimate.
dim_theta = size(X_entry_joint,2);
% Set starting value for theta: very robust to starting values, just make sure to not set them too high for fminunc to work well.       
theta_0 = 0.01 * randn(dim_theta,1);
% Construct anonymous function to pass to minimzers.
min_neg_ll = @(theta) neg_log_ll_entry(theta,N_joint-1,X_entry_joint,N_pot_joint);
% Check whether likelihood is defined at starting value.
min_neg_ll(theta_0);
% Use gradient-based fminunc.
[theta_opt_entry_joint, neg_ll_opt_entry_joint] = fminunc(min_neg_ll, theta_0,fminunc_options);
% Save vector of coefficient estimates to file.
save(project_paths('OUT_ANALYSIS','joint_entry_q_parameters'),'theta_opt_entry_joint');
save(project_paths('OUT_ANALYSIS','joint_entry_char'),'X_entry_joint');
save(project_paths('OUT_ANALYSIS','joint_entry_npot'),'N_pot_joint');


%% Compute probabilities for each number of bidders (only entrants are counted) for each auction based on parameter estimates for entry probability above.
N_obs = size(X_entry_joint,1);
E_N = zeros(N_obs,1); % expected number of entrants for each auction.
prob_N_grid = zeros(N_obs, N_pot_max_joint);
prob_N_zero = zeros(N_obs,1);
% Vector for probability of individual entrant entering.
prob_q_joint = zeros(N_obs,1);
% Loop over auctions.
for t=1:N_obs
    % Compute probability of an individual entrant entering the auction.
    q_prob_num = exp(X_entry_joint(t,:) * theta_opt_entry_joint);
    prob_q_joint(t,1) = q_prob_num ./ (1.0 + q_prob_num); 
    % Compute probability of no entrant entering auction t.
    prob_N_zero(t) = nchoosek(N_pot_joint(t),0) .* prob_q_joint(t).^(0) .* (1-prob_q_joint(t)).^(N_pot_joint(t));
    % Loop over potential number of (entrant) bidders.
    for n=1:N_pot_joint(t)
        % Compute probability of exactly n bidders entering auction t.
        prob_N_grid(t,n) = nchoosek(N_pot_joint(t),n) .* prob_q_joint(t).^(n) .* (1-prob_q_joint(t)).^(N_pot_joint(t)-n);
    end
    % Compute expected number of bidders for auction t.
    E_N(t) = sum(prob_N_grid(t,1:N_pot_joint(t)) .* linspace(1,N_pot_joint(t),N_pot_joint(t)));
end
fprintf('Expected and observed number of entrants (average across all auctions): %6.4f vs %6.4f\n',mean(E_N),mean(N_joint-1));

% Split predicted number of bidders up into gross and net.
N_gross = N_obs-sum(X_entry_joint(:,2))
N_net_vec = N_net;
N_net = N_obs-N_gross
fprintf('Expected and observed number of entrants (average across all gross auctions): %6.4f vs %6.4f\n',mean(E_N(1:N_gross)),mean(N_joint(1:N_gross)-1));
fprintf('Expected and observed number of entrants (average across all net auctions): %6.4f vs %6.4f\n',mean(E_N(N_gross+1:end)),mean(N_joint(N_gross+1:end)-1));

%Compare expected against realized number of bidders.
N_comp = [N_joint-1, E_N];
subplot(1,2,1)
hist(N_comp)
title('Distribution of realized and expected number of entrants');
legend('Realized N entrants','Expected N entrants');
subplot(1,2,2)
hist(N_comp(:,1)-E_N)
title('Distribution of difference between realized and expected number of entrants');

% Check whether probability of n bidders entering sum up to one.
check_prob_N = sum(prob_N_grid,2);
disp('---BEGIN CHECKING OF PLAUSIBIILITY OF NUMBER OF BIDDERS ENTERING---');
fprintf('Average probability (across net auctions) of at least one entrant entering in the auction is %0.4f.\n', mean(check_prob_N))
% Check whether probability of n bidders entering sum up to one when zero (no entrant entering is included).
check_prob_N_total = sum(prob_N_grid,2) + prob_N_zero;
fprintf('On average probabilities of all potential n (including no entrant entering, across net auctions) sum up to %0.4f.\n', mean(check_prob_N_total))
fprintf('Minimum of sum over potential bidders is %0.4f.\n', min(check_prob_N_total))
fprintf('Maximum of sum over potential bidders is %0.4f.\n', max(check_prob_N_total))
fprintf('Average probability of no entrant entering the auction is %0.4f.\n', mean(prob_N_zero))
fprintf('Minimum probability (across net auctions) of no entrant entering is %0.4f.\n', min(prob_N_zero))
fprintf('Maximum probability (across net auctions) of no entrant entering is %0.4f.\n', max(prob_N_zero))
disp('---END OF CHECKING PLAUSIBIILITY OF NUMBER OF BIDDERS ENTERING---');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Predict counterfactual number of bidders when going form net to gross.
E_N_cfnetgross = zeros(N_net,1); % expected number of entrants for each auction.
fprintf('Mean and median (across net auctions) of predicted total bidders when net procured as gross:\n%1.1f and %1.1f.\n', 1+mean(E_N_cfnetgross),1+median(E_N_cfnetgross));
fprintf('Mean and median (across gross auctions) of predicted total bidders in gross auctions:\n%1.1f and %1.1f.\n', 1+mean(E_N(1:N_gross)),1+median(E_N(1:N_gross)));
fprintf('Mean and median (across net auctions) of predicted total bidders in net auctions:\n%1.1f and %1.1f.\n', 1+mean(E_N(N_gross+1:end)),1+median(E_N(N_gross+1:end)));

prob_N_cf_grid = zeros(N_net, N_pot_max_net);
prob_N_cf_zero = zeros(N_net,1);
% Vector for probability of individual entrant entering.
prob_q_cf = zeros(N_net,1);
% Loop over auctions.
for t=1:N_net
    % Simulate going from net to gross.
    X_entry_joint(N_gross+t,2) = 0;
    % Compute probability of an individual entrant entering the auction.
    q_prob_num = exp(X_entry_joint(t,:) * theta_opt_entry_joint);
    prob_q_cf(t,1) = q_prob_num ./ (1.0 + q_prob_num); 
    % Compute probability of no entrant entering auction t.
    prob_N_cf_zero(t) = nchoosek(N_pot_joint(t),0) .* prob_q_joint(t).^(0) .* (1-prob_q_joint(t)).^(N_pot_joint(t));
    % Loop over potential number of (entrant) bidders.
    for n=1:N_pot_joint(t)
        % Compute probability of exactly n bidders entering auction t.
        prob_N_cf_grid(t,n) = nchoosek(N_pot_joint(t),n) .* prob_q_joint(t).^(n) .* (1-prob_q_joint(t)).^(N_pot_joint(t)-n);
    end
    % Compute expected number of bidders for auction t.
    E_N_cfnetgross(t) = sum(prob_N_cf_grid(t,1:N_pot_joint(t)) .* linspace(1,N_pot_joint(t),N_pot_joint(t)));
end
fprintf('Expected and observed number of entrants (average across all counterfactual auctions net->gross): %6.4f vs %6.4f\n',mean(E_N_cfnetgross),mean(N_net_vec-1));
% Check whether probability of n bidders entering sum up to one.
check_prob_N_cf = sum(prob_N_cf_grid,2);
disp('---BEGIN CHECKING OF PLAUSIBIILITY OF NUMBER OF BIDDERS ENTERING in COUNTERFACTUAL---');
fprintf('Average probability (across net auctions procured as gross) of at least one entrant entering in the auction is %0.4f.\n', mean(check_prob_N_cf))
% Check whether probability of n bidders entering sum up to one when zero (no entrant entering is included).
check_prob_N_cf_total = sum(prob_N_cf_grid,2) + prob_N_cf_zero;
fprintf('On average probabilities of all potential n (including no entrant entering, across net auctions) sum up to %0.4f.\n', mean(check_prob_N_cf_total))
fprintf('Minimum of sum over potential bidders is %0.4f.\n', min(check_prob_N_cf_total))
fprintf('Maximum of sum over potential bidders is %0.4f.\n', max(check_prob_N_cf_total))
fprintf('Average probability of no entrant entering the auction is %0.4f.\n', mean(prob_N_zero))
fprintf('Minimum probability (across net auctions) of no entrant entering is %0.4f.\n', min(prob_N_cf_zero))
fprintf('Maximum probability (across net auctions) of no entrant entering is %0.4f.\n', max(prob_N_cf_zero))
disp('---END OF CHECKING PLAUSIBIILITY OF NUMBER OF BIDDERS ENTERING IN COUNTERFACTUAL---');

%% Export probabilities for each entrant bidder configuration.
prob_N_grid_gross = prob_N_grid(1:N_gross,:);
prob_N_grid_net = prob_N_grid(N_gross+1:end,:);

% Save data and full workspace for reuse in entry cost estimation.
save(project_paths('OUT_ANALYSIS','ga_entry_n_probs'),'prob_N_grid_gross');
save(project_paths('OUT_ANALYSIS','na_entry_n_probs'),'prob_N_grid_net');
save(project_paths('OUT_ANALYSIS','ga_entry_npot'),'N_pot_gross');
save(project_paths('OUT_ANALYSIS','na_entry_npot'),'N_pot_net');
save(project_paths('OUT_ANALYSIS','cfnetgross_entry_n_probs'),'prob_N_cf_grid');
save(project_paths('OUT_ANALYSIS','entry_probs_joint_workspace.mat'));