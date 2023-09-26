"""
    Write the results for counterfactual efficiency probabilities and revenues as table code into a .tex-file.

"""
import numpy as np
import pandas as pd
from bld.project_paths import project_paths_join

# Load estimation results for gross and net sample from csv-file.
cf_eff_data = pd.read_csv(project_paths_join('OUT_ANALYSIS', 'cf_eff_data.csv'))
cf_eff_data_cons_n = pd.read_csv(project_paths_join('OUT_ANALYSIS', 'cf_eff_data_cons_n.csv'))
cf_2_eff_data = pd.read_csv(project_paths_join('OUT_ANALYSIS', 'cf_2_eff_data.csv'))

# Load correct file for either using mean or median revenue statistics.
cf_rev_data = pd.read_csv(project_paths_join('OUT_ANALYSIS', 'cf_rev_data.csv'))
cf_rev_data_cons_n = pd.read_csv(project_paths_join('OUT_ANALYSIS', 'cf_rev_data_cons_n.csv'))

auction_mode_list = ['Gross auctions (endo.~N)','Net auctions (endo.~N)','Net $\\rightarrow$ Gross auctions (endo.~N)']
auction_mode_list_cons_n = ['Gross auctions (fixed N)','Net auctions (fixed N)','Net $\\rightarrow$ Gross auctions (fixed N)']
# Write results for efficiency comparison to LaTeX table.
with open(project_paths_join('OUT_TABLES', 'cfeffclean.tex'), 'w') as tex_file:
    # Set a different types of table row with placeholders.
    table_row_header = ' & {val1} & {val2} & {val3} \\tabularnewline \n'
    table_row_stat = '{stat} & {val1:.4f} & {val2:.4f} & {val3:.4f} \\tabularnewline\n'

    # Top of table.
    tex_file.write('\\begin{tabular}{rccc}\n\\toprule\n')
    # Header row with names.
    tex_file.write(table_row_header.format(val1='Gross auctions',val2='Net auctions', val3='Net $\\rightarrow$ Gross'))
    tex_file.write('\\midrule \n')
    # Write efficiency probabilities.
    # Row 0 indicates the mean efficiency probabilities, Row 1 uses median.
    tex_file.write(table_row_stat.format(stat='Endogenous $N$',
                                         val1=cf_eff_data['gross'][0], val2=cf_eff_data['net'][0],
                                         val3=cf_eff_data['cfnetgross'][0]))
    tex_file.write(table_row_stat.format(stat='Observed $N$',
                                         val1=cf_eff_data_cons_n['gross'][0], val2=cf_eff_data_cons_n['net'][0],
                                         val3=cf_eff_data_cons_n['cfnetgross'][0]))
    tex_file.write('\\midrule  \n')
    # Add notes to table.
    tex_file.write('\\multicolumn{4}{p{12cm}}{\\footnotesize{\\textit{Notes: The table displays the mean (across auctions) of the ex ante probabilities of selecting the efficient firm in our two different samples (gross and net) and a counterfactual scenario in which the net auction sample is procured using gross auctions. The first row indicates efficiency probabilities when the number of bidders is endogenous and determined by our entry model. The second row indicates efficiency probabilities when the number of bidders is fixed and as observed in the data. }}}\\tabularnewline\n')
    # Bottom of table.
    tex_file.write('\\bottomrule\n\\end{tabular}\n')
    
    
# Write results for efficiency comparison for different ways of procuring net sample.
with open(project_paths_join('OUT_TABLES', 'cf2effclean.tex'), 'w') as tex_file:
    # Set a different types of table row with placeholders.
    table_row_header = ' & {val1} & {val2} & {val3} \\tabularnewline \n'
    table_row_stat = '{stat} & {val1:.4f} & {val2:.4f} & {val3:.4f} \\tabularnewline\n'

    # Top of table.
    tex_file.write('\\begin{tabular}{rccc}\n\\toprule\n')
    # Header row with names.
    tex_file.write(table_row_header.format(val3='Net auctions',val2='Net auctions', val1='Net auctions'))
    tex_file.write(table_row_header.format(val3='(observed)',val2='(w/ gross markups)', val1='(non-strategic)'))
    tex_file.write('\\midrule \n')
    # Write efficiency probabilities.
    # Row 0 indicates the mean efficiency probabilities, Row 1 uses median.
    tex_file.write(table_row_stat.format(stat='Pr(selecting efficient firm)',
                                         val3=cf_eff_data['net'][0], val2=cf_2_eff_data['gross_mu'][0],
                                         val1=cf_2_eff_data['non_strategic'][0]))
    tex_file.write('\\midrule  \n')
    # Add notes to table.
    tex_file.write('\\multicolumn{4}{p{13.5cm}}{\\footnotesize{\\textit{Notes: The table displays the mean (across auctions) of the ex ante probabilities of selecting the efficient firm in the net auction sample for different hypothetical scenarios. For the simulations in this table, we assume that the number of bidders is fixed at the observed level.}}}\\tabularnewline\n')
    # Bottom of table.
    tex_file.write('\\bottomrule\n\\end{tabular}\n')

# Write results for revenue and agency payoff comparison to LaTeX table.
# Version with endogenous number of bidders. NEW
with open(project_paths_join('OUT_TABLES', 'cfrevclean.tex'), 'w') as tex_file:
    # Set a different types of table row with placeholders.
    table_row_header = ' & {val1} & {val2} & {val3} & {val4} \\tabularnewline \n'
    table_row_stat = '{stat} & {val1:.4f} & {val2:.4f} & {val3:.4f} &{val4:.4f} \\tabularnewline\n'

    # Top of table.
    tex_file.write('\\begin{tabular}{rcccc}\n\\toprule\n')
    # Header row with names and dependent variables.
    tex_file.write(table_row_header.format(val1='Observed bid',val2='Predicted bid',
                                           val3='E(ticket revenues)', val4='E(agency payoff)'))
    tex_file.write('\\midrule \n')
    # Write revenue results.
    for idx, val in enumerate(auction_mode_list):
        # This uses means of the revenue statistics.
        tex_file.write(table_row_stat.format(stat=val,
                                         val1=cf_rev_data['bid_obs'][idx], val2=cf_rev_data['bid_pred'][idx],
                                         val3=cf_rev_data['mean_rev'][idx], val4=cf_rev_data['e_payoff'][idx]))

 

    tex_file.write('\\midrule  \n')
    # Add notes to table.
    tex_file.write('\\multicolumn{5}{p{14.75cm}}{\\footnotesize{\\textit{Notes: The table summarizes the observed and predicted bids along with the median of the expected ticket revenues and expected procurement agency payoff for our two different samples (net and gross) and a counterfactual scenario in which the net auction sample is procured using gross auctions. Statistics integrate over the distribution of predicted entry behavior, i.e., the number of bidders is endogenous. For the gross auction sample, the net auction sample, and the net auctions procured as gross counterfactual we predict on average 4.2, 3.3, and 4.4 bidders, respectively, when the number of bidders is endogenous.}}}\\tabularnewline\n')
    # Bottom of table.
    tex_file.write('\\bottomrule\n\\end{tabular}\n')

# Write results for revenue and agency payoff comparison to LaTeX table.
# Version with exogenous/fixed number of bidders. OLF
with open(project_paths_join('OUT_TABLES', 'cfrevcleansmallconsn.tex'), 'w') as tex_file:
    # Set a different types of table row with placeholders.
    table_row_header = ' & {val1} & {val2} & {val3} \\tabularnewline \n'
    table_row_stat = '{stat} & {val1:.4f} & {val2:.4f} & {val3:.4f} \\tabularnewline\n'

    # Top of table.
    tex_file.write('\\begin{tabular}{rccc}\n\\toprule\n')
    # Header row with names and dependent variables.
    tex_file.write(table_row_header.format(val1='Predicted winning bid',
                                           val2='E(ticket revenues)', val3='E(agency payoff)'))
    tex_file.write('\\midrule \n')
    # Write revenue results.
    for idx, val in enumerate(auction_mode_list):
        # This uses means of revenue statistics.
        tex_file.write(table_row_stat.format(stat=val,
                                         val1=cf_rev_data_cons_n['bid_pred'][idx],
                                         val2=cf_rev_data_cons_n['mean_rev'][idx], val3=cf_rev_data_cons_n['e_payoff'][idx]))
    tex_file.write('\\midrule  \n')
    # Add notes to table.
    tex_file.write('\\multicolumn{4}{p{14.75cm}}{\\footnotesize{\\textit{Notes: The table summarizes the predicted winning bids along with the mean of the expected ticket revenues and expected procurement agency payoff for our two different samples (net and gross) and a counterfactual scenario in which the net auction sample is procured using gross auctions. Means are calculated over all auctions within the respective samples, i.e., gross and net. The number of bidders is held constant as observed in our data.}}}\\tabularnewline\n')
    # Bottom of table.
    tex_file.write('\\bottomrule\n\\end{tabular}\n')


# Write results for revenue and agency payoff comparison to LaTeX table.
# We both endogenous N and fixed N in one table.
with open(project_paths_join('OUT_TABLES', 'cfrevcleansmallcombined.tex'), 'w') as tex_file:
    # Set a different types of table row with placeholders.
    table_row_header = ' & {val1} & {val2} & {val3} \\tabularnewline \n'
    table_row_stat = '{stat} & {val1:.4f} & {val2:.4f} & {val3:.4f} \\tabularnewline\n'

    # Top of table.
    tex_file.write('\\begin{tabular}{rccc}\n\\toprule\n')
    # Header row with names and dependent variables.
    tex_file.write(table_row_header.format(val1='Predicted winning bid',
                                           val2='E(ticket revenues)', val3='E(agency payoff)'))
    tex_file.write('\\midrule \n')
    # Write revenue results.
    for idx, val in enumerate(auction_mode_list):
        # This uses means of revenue statistics.
        tex_file.write(table_row_stat.format(stat=val,
                                         val1=cf_rev_data['bid_pred'][idx],
                                         val2=cf_rev_data['mean_rev'][idx], val3=cf_rev_data['e_payoff'][idx]))
    tex_file.write('\\midrule  \n')
    for idx, val in enumerate(auction_mode_list_cons_n):
        # This uses means of revenue statistics.
        if idx==2: # only print this for counterfactual to save space.
            tex_file.write(table_row_stat.format(stat=val,
                                            val1=cf_rev_data_cons_n['bid_pred'][idx],
                                            val2=cf_rev_data_cons_n['mean_rev'][idx], val3=cf_rev_data_cons_n['e_payoff'][idx]))
    tex_file.write('\\midrule  \n')
    # Add notes to table.
    tex_file.write('\\multicolumn{4}{p{16.5cm}}{\\footnotesize{\\textit{Notes: The table summarizes the predicted winning bids along with the mean of the expected ticket revenues and expected procurement agency payoff for our two different samples (net and gross) and a counterfactual scenario in which the net auction sample is procured using gross auctions. Means are calculated over all auctions within the respective samples, i.e., gross and net. The upper panel displays statistics when the number of bidders is endogenous as predicted by our entry model. The lower panels displays statistics when the number of bidders is fixed at the values observed in the data. For the gross auction sample, the net auction sample, and the net auctions procured as gross counterfactual we predict on average 4.2, 3.3, and 4.4 bidders, respectively, when the number of bidders is endogenous.}}}\\tabularnewline\n')
    # Bottom of table.
    tex_file.write('\\bottomrule\n\\end{tabular}\n')