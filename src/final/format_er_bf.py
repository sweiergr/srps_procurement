"""
    Write the results of the bid function estimation as table code into a .tex-file.

"""
import numpy as np
import pandas as pd
from bld.project_paths import project_paths_join

# Load estimation results for gross and net sample from csv-file.
er_gross = pd.read_csv(project_paths_join('OUT_ANALYSIS', 'erbf_gross.csv'),header=None)
legend_gross = pd.read_csv(project_paths_join('OUT_ANALYSIS', 'erbf_gross_legend.csv'),header=None,na_values=[''])
legend_gross.drop(legend_gross.columns[[-1]], axis=1, inplace=True)
legend_gross = legend_gross.fillna('')
er_gross.columns=[u'beta',u'sigma',u'tstat',u'pval']

er_net = pd.read_csv(project_paths_join('OUT_ANALYSIS', 'erbf_net.csv'),header=None)
legend_net = pd.read_csv(project_paths_join('OUT_ANALYSIS', 'erbf_net_legend.csv'),header=None)
legend_net.drop(legend_net.columns[[-1]], axis=1, inplace=True)
legend_net = legend_net.fillna('')
er_net.columns=[u'beta',u'sigma',u'tstat',u'pval']


# Set a different types of table row with placeholders.
table_row_var = """\multirow{{2}}{{*}}{{{var}}} & {val1:.4f}{str1} & {val2:.2f} \\tabularnewline\n  & ({se1:.4f}) & \\tabularnewline\n"""
table_row_var_2 = """\multirow{{2}}{{*}}{{{var}}} & {val2:.4f}{str2} & {val3:.4f}{str3}  \\tabularnewline\n  & ({se2:.4f}) & ({se3:.4f}) \\tabularnewline\n"""
table_row_strings = '{stat} & {val1} & {val2} \\tabularnewline \n'

# Add significance stars.
er_gross['sig_stars'] = ''
er_gross.loc[er_gross.pval>0.1, 'sig_stars'] = '' 
er_gross.loc[(er_gross.pval<=0.1) & (er_gross.pval>0.05), 'sig_stars'] = '*' 
er_gross.loc[(er_gross.pval<=0.05) & (er_gross.pval>0.01), 'sig_stars'] = '**' 
er_gross.loc[er_gross.pval<=0.01,'sig_stars'] = '***' 
er_net['sig_stars'] = ''
er_net.loc[er_net.pval>0.1, 'sig_stars'] = '' 
er_net.loc[(er_net.pval<=0.1) & (er_net.pval>0.05), 'sig_stars'] = '*' 
er_net.loc[(er_net.pval<=0.05) & (er_net.pval>0.01), 'sig_stars'] = '**' 
er_net.loc[er_net.pval<=0.01,'sig_stars'] = '***' 

# Write the results to a LaTeX table.
with open(project_paths_join('OUT_TABLES', 'estresultsbf.tex'), 'w') as tex_file:

    # Top of table.
    tex_file.write('\\begin{tabular}{lcc}\n\\toprule\n')
    # Header row with names and dependent variables.
    tex_file.write(table_row_strings.format(stat='', val1='Gross auctions',val2='Net auctions'))
   
    tex_file.write('\\midrule')

    # Write beta-coefficients to table with 2 models.
    for i, var in enumerate(legend_gross):
        tex_file.write(table_row_var_2.format(var=legend_gross.at[0,i],
                                            val2=er_gross['beta'][i],
                                            str2=er_gross['sig_stars'][i],
                                            se2=er_gross['sigma'][i],
                                            val3=er_net['beta'][i],
                                            str3=er_net['sig_stars'][i],
                                            se3=er_net['sigma'][i]))
    
    tex_file.write('\\midrule\n')
    tex_file.write('\\multicolumn{3}{p{6cm}}{\\footnotesize{\\textit{Notes: MLE-SE in parentheses. *,**,*** denote significance at the 10\%, 5\% and 1\%-level respectively.}}}\\tabularnewline\n')
    # Bottom of table.
    tex_file.write('\\bottomrule\n\\end{tabular}\n')