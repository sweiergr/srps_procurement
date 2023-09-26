%{
    Write estimate for alpha parameters to LaTeX table.	

%}

% Load workspace from second stage GMM estimation.
load(project_paths('OUT_ANALYSIS','postestimation_workspace_net.mat'));
sig_stars_legend = {'','*','**','***'};
sig_stars_matrix = 1 .* (p_alpha_grid>0.1 | isnan(p_alpha_grid)) + 2 .* (p_alpha_grid<=0.1 & p_alpha_grid>0.05) + 3 .* (p_alpha_grid<=0.05 & p_alpha_grid>0.01) +  4 .* (p_alpha_grid<0.01);
sig_stars = sig_stars_legend(sig_stars_matrix);

%% Add information on estimated revenue signal distribution.
% Load auxiliary data on revenue in gross auctions.
load(project_paths('OUT_ANALYSIS','rev_est_gross'));


mean_rev_gross = mean(E_TR_gross);
mean_rev_net = mean(E_TR_net);
sd_rev_net = mean(sigma_revenue);
sd_rev_gross = mean(sigma_rev_gross);
rev_per_zkm_gross = mean(rev_per_zkm_gross);
rev_per_zkm_net = mean(rev_per_zkm);
fprintf('Average (median) revenue per zkm: EUR %4.2f (%4.2f).\n',mean([rev_per_zkm_gross;rev_per_zkm]),median([rev_per_zkm_gross;rev_per_zkm]));
% Open file handle to write table to..
fid = fopen(project_paths('OUT_TABLES','estresultsalpha.tex'),'w');
fprintf(fid,'\\begin{tabular}{r|cccc} \\toprule \n')
fprintf(fid, '& $N=2$ & $N=3$ & $N=4$ & $N=5$ \\\\ \n');
fprintf(fid,'\\midrule \n')
% Row for alpha_I.
fprintf(fid, '%8.10s & $%8.4f$%s & $%8.4f$%s & $%8.4f$%s & $%8.4f$%s \\\\ \n ', '$\alpha_I$', alpha_I_grid(1,1), sig_stars{1,1},  alpha_I_grid(1,2), sig_stars{1,2},  alpha_I_grid(1,3), sig_stars{1,3},  alpha_I_grid(1,4), sig_stars{1,4});   
fprintf(fid, '		& $(%8.4f)$ & $(%8.4f)$ & $(%8.4f)$ & $(%8.4f)$ \\\\ \n', alpha_SE(1,1), alpha_SE(1,2), alpha_SE(1,3), alpha_SE(1,4));
% Row for alpha_E.
fprintf(fid, '%8.10s & $%8.4f$%s & $%8.4f$%s & $%8.4f$%s & $%8.4f$%s \\\\ \n ', '$\alpha_E$', alpha_E_grid(1,1), sig_stars{2,1},  alpha_E_grid(1,2), sig_stars{2,2},  alpha_E_grid(1,3), sig_stars{2,3},  alpha_E_grid(1,4), sig_stars{2,4});   
fprintf(fid, '      & $(%8.4f)$ & $(%8.4f)$ & $(%8.4f)$ & $(%8.4f)$ \\\\ \n', alpha_SE(2,1), alpha_SE(2,2), alpha_SE(2,3), alpha_SE(2,4));
% Row for informational advantage.
fprintf(fid, '\\midrule \n ');
% fprintf(fid, '%8.28s & $%8.4f$ & $%8.4f$ & $%8.4f$ & $%8.4f$ \\\\ \n ', '$\frac{\alpha_I}{\alpha_E}$', info_advantage(1,1), info_advantage(1,2), info_advantage(1,3), info_advantage(1,4))   
fprintf(fid, '%8.78s & $%8.4f$ & $%8.4f$ & $%8.4f$ & $%8.4f$ \\\\ \n ', '$var_E\left[R|r_E=r\right]/var_I\left[R|r_I=r\right]$', ratio_residual_variance(1,1), ratio_residual_variance(1,2), ratio_residual_variance(1,3), ratio_residual_variance(1,4));   
fprintf(fid, '%8.78s & $%8.4f$ & $%8.4f$& & \\\\ \n ', 'Mean \& SD rev.~signal, gross', mean_rev_gross, sd_rev_gross);   
fprintf(fid, '%8.78s & $%8.4f$ &$%8.4f$  & & \\\\ \n ', 'Mean \& SD rev.~signal, net', mean_rev_net, sd_rev_net);   
%fprintf(fid, '%8.78s & $%8.4f$ & & & \\\\ \n ', 'SD(revenue signal)', sd_rev);   
fprintf(fid, '\\midrule \n ');
% Write bottom rule
fprintf(fid, '\\midrule \n ');
fprintf(fid,'\\multicolumn{5}{p{13.25cm}}{\\footnotesize{\\textit{Notes: The table displays the estimated asymmetry parameters for the incumbent (row $\\alpha_I$) and the entrants (row $\\alpha_E$) for different numbers of participating bidders. Parameters are estimated using maximum likelihood. Standard errors are computed using the delta method. *,**,*** denote significance at the 10, 5 and 1 percent-level for testing $H_0: \\alpha=\\frac{1}{N}$, respectively. Row 4 and 5 display the estimated mean and standard deviation (SD) of the revenue signal distribution (in 10-million euros). Row 5 averages the estimated statistics across all net auctions. Row 4 computes the analogous hyothetical revenue signal statistics for the gross auction sample based on the estimated revenue signal parameters from the net auction estimation and the observed contract characteristics of the gross auction sample.}}}\\\\ \n');
fprintf(fid, '\\bottomrule \n');
% Close tabular part.
fprintf(fid,  '\\end{tabular}\n');
% Close file handle.
fclose(fid);