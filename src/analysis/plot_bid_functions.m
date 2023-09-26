function plot_bid_functions(lambda_I, lambda_E, rho_I, rho_E, plot_title, filename, b_range)
    global N_obs
    % Plot estimated bid functions.
    
    % Compute parameters for grid.
    n_bid_grid = 2000;
    lambda_I_grid = repmat(lambda_I,1,n_bid_grid);
    lambda_E_grid = repmat(lambda_E,1,n_bid_grid);
    rho_I_grid = repmat(rho_I,1,n_bid_grid);
    rho_E_grid = repmat(rho_E,1,n_bid_grid);
    bid_grid = repmat(linspace(b_range(1),b_range(2),n_bid_grid),N_obs,1);
    % Compute estimated bid distributions.
    bid_distribution_I = ones(N_obs,n_bid_grid) - exp( - (bid_grid ./ lambda_I_grid) .^ (rho_I_grid) ); 
    bid_distribution_E = ones(N_obs,n_bid_grid) - exp( - (bid_grid ./ lambda_E_grid) .^ (rho_E_grid) ); 
    bid_density_I = exp( - (bid_grid ./ lambda_I_grid)) .^ (rho_I_grid-1) .* (bid_grid ./ lambda_I_grid);
    bid_density_E = exp( - (bid_grid ./ lambda_E_grid)) .^ (rho_E_grid-1) .* (bid_grid ./ lambda_E_grid);

    % Plot estimated bid distributions for each contract.
    for i=1:N_obs
        clear title xlabel ylabel
        subplot(1,1,1)
        plot(bid_grid(i,:), bid_distribution_I(i,:), bid_grid(i,:), bid_distribution_E(i,:));
        graphtitle = sprintf(strcat(plot_title ,num2str(i)));
        title(graphtitle);
        legend('Incumbent', 'Entrant')
        fname = sprintf(regexprep(strcat(project_paths('OUT_FIGURES',filename),num2str(i)),'\','\\\'));
        saveas(gcf,fname,'pdf');
    end
end
