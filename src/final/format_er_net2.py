"""
    Write the results of the second step net auction estimation as table code into a .tex-file.

"""
import pdb
import numpy as np
import pandas as pd
from bld.project_paths import project_paths_join

# Load estimation results for gross and net sample from csv-file.
er_net2 = pd.read_csv(project_paths_join('OUT_ANALYSIS', 'er_net2.csv'),header=None)
legend_net2 = pd.read_csv(project_paths_join('OUT_ANALYSIS', 'er_net2_legend.csv'),header=None,na_values=[''])
legend_net2.drop(legend_net2.columns[[-1]], axis=1, inplace=True)
legend_net2 = legend_net2.fillna('')
er_net2.columns=[u'beta',u'sigma',u'tstat',u'pval']

# Set a different types of table row with placeholders.
table_row_var = """\multirow{{2}}{{*}}{{{var}}} & {val1:.4f}{str1} \\tabularnewline\n  & ({se1:.4f}) \\tabularnewline\n"""
table_row_var_2 = """\multirow{{2}}{{*}}{{{var}}} & {val2:.4f}{str2} & {val3:.4f}{str3}  \\tabularnewline\n  & ({se2:.4f}) & ({se3:.4f}) \\tabularnewline\n"""
table_row_strings = '{stat} & {val1} \\tabularnewline \n'
er_net2['sig_stars'] = ''
er_net2.loc[er_net2.pval>0.1, 'sig_stars'] = '' 
er_net2.loc[(er_net2.pval<=0.1) & (er_net2.pval>0.05), 'sig_stars'] = '*' 
er_net2.loc[(er_net2.pval<=0.05) & (er_net2.pval>0.01), 'sig_stars'] = '**' 
er_net2.loc[er_net2.pval<=0.01,'sig_stars'] = '***' 


# Write the results to a LaTeX table.
with open(project_paths_join('OUT_TABLES', 'estresultsnet2.tex'), 'w') as tex_file:

    # Top of table.
    tex_file.write('\\begin{tabular}{lcc}\n\\toprule\n')
    # Header row with names and dependent variables.
    tex_file.write(table_row_strings.format(stat='', val1='Revenue parameter estimates'))
    
    tex_file.write('\\midrule')
    # Write coefficients to table.
    for i, var in enumerate(legend_net2):
        tex_file.write(table_row_var.format(var=legend_net2.at[0,i],
                                            val1=er_net2['beta'][i],
                                            str1=er_net2['sig_stars'][i],
                                            se1=er_net2['sigma'][i]))

    # Write middle part of the table including table note.
    tex_file.write('\\midrule\n')
    tex_file.write('\\multicolumn{2}{p{6cm}}{\\footnotesize{\\textit{Notes: MLE-SE in parentheses. *,**,*** denote significance at the 10\%, 5\% and 1\%-level respectively.}}}\\tabularnewline\n')
    # Bottom of table.
    tex_file.write('\\bottomrule\n\\end{tabular}\n')