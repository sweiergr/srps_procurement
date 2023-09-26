%{ 
    Test code to simulate entry probabilities and check that they sum up to one.

%}

%% Define parameters and containers.
% Prob of specific player entering.
q = 0.5;
% Number of potential bidders.
N_pot = 10;
% Probabilities of N bidders in auction.
prob_entry = zeros(N_pot,1);
%% Compute actual entry probabilities.
% Loop over all potential bidder configurations entering.
for n=1:N_pot
    % Prob of N bidders entering.
    pr_entry_n = nchoosek(N_pot,n) .* q^n .* (1-q)^(N_pot-n);
    % Write to container.
    prob_entry(n) = pr_entry_n;
end
%% Print results of interest.
fprintf('Sum of all entry probabilities is %0.4f.\n',sum(prob_entry));