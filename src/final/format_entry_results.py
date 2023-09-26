"""
    Write the results for estimated entry costs as table code into a .tex-file.

"""
import numpy as np
import pandas as pd
from bld.project_paths import project_paths_join

# Load entry estimation results for gross and net sample from csv-file.
ed_gross = pd.read_csv(project_paths_join('OUT_ANALYSIS', 'entry_results_gross.csv'),header=None)
ed_net = pd.read_csv(project_paths_join('OUT_ANALYSIS', 'entry_results_net.csv'),header=None)
# Set column names.
col_names = [u'year',u'n_bidder',u'bid_win',u'kappa']
ed_gross.columns = col_names
ed_net.columns = col_names

# FOR GROSS SAMPLE.
# Compute relevant statistics in million EUR.
kappa_mean_gross = 10 * ed_gross['kappa'].mean()
kappa_median_gross = 10 * ed_gross['kappa'].median()
kappa_std_gross = 10 * ed_gross['kappa'].std()
# Compute entry cost as share of winning bid.
kappa_relative_gross = ed_gross['kappa'] / ed_gross['bid_win']
kappa_rel_mean_gross = 10 * kappa_relative_gross.mean()
kappa_rel_median_gross = 10 * kappa_relative_gross.median()
kappa_rel_std_gross = 10 * kappa_relative_gross.std()

# Construct early and late sample.
# EARLY GROSS SAMPLE.
ed_gross_early = ed_gross.loc[ed_gross['year']<=2004]
# Compute statistics for early sample.
kappa_mean_gross_early = 10 * ed_gross_early['kappa'].mean()
kappa_median_gross_early = 10 * ed_gross_early['kappa'].median()
kappa_std_gross_early = 10 * ed_gross_early['kappa'].std()
# Compute entry cost as share of winning bid.
kappa_relative_gross_early = ed_gross_early['kappa'] / ed_gross_early['bid_win']
kappa_rel_mean_gross_early = 10 * kappa_relative_gross_early.mean()
kappa_rel_median_gross_early = 10 * kappa_relative_gross_early.median()
kappa_rel_std_gross_early = 10 * kappa_relative_gross_early.std()
# LATE GROSS SAMPLE.
ed_gross_late = ed_gross.loc[ed_gross['year']>2004]
# Compute statistics for early sample.
kappa_mean_gross_late = 10 * ed_gross_late['kappa'].mean()
kappa_median_gross_late = 10 * ed_gross_late['kappa'].median()
kappa_std_gross_late = 10 * ed_gross_late['kappa'].std()
# Compute entry cost as share of winning bid.
kappa_relative_gross_late = ed_gross_late['kappa'] / ed_gross_late['bid_win']
kappa_rel_mean_gross_late = 10 * kappa_relative_gross_late.mean()
kappa_rel_median_gross_late = 10 * kappa_relative_gross_late.median()
kappa_rel_std_gross_late = 10 * kappa_relative_gross_late.std()

# FOR NET SAMPLE.
# Compute relevant statistics in million EUR.
kappa_mean_net = 10 * ed_net['kappa'].mean()
kappa_median_net = 10 * ed_net['kappa'].median()
kappa_std_net = 10 * ed_net['kappa'].std()
# Compute entry cost as share of winning bid.
kappa_relative_net = ed_net['kappa'] / ed_net['bid_win']
kappa_rel_mean_net = 10 * kappa_relative_net.mean()
kappa_rel_median_net = 10 * kappa_relative_net.median()
kappa_rel_std_net = 10 * kappa_relative_net.std()

# Construct early and late sample.
# EARLY NET SAMPLE.
ed_net_early = ed_net.loc[ed_net['year']<=2004]
# Compute statistics for early sample.
kappa_mean_net_early = 10 * ed_net_early['kappa'].mean()
kappa_median_net_early = 10 * ed_net_early['kappa'].median()
kappa_std_net_early = 10 * ed_net_early['kappa'].std()
# Compute entry cost as share of winning bid.
kappa_relative_net_early = ed_net_early['kappa'] / ed_net_early['bid_win']
kappa_rel_mean_net_early = 10 * kappa_relative_net_early.mean()
kappa_rel_median_net_early = 10 * kappa_relative_net_early.median()
kappa_rel_std_net_early = 10 * kappa_relative_net_early.std()
# LATE NET SAMPLE.
ed_net_late = ed_net.loc[ed_net['year']>2004]
# Compute statistics for early sample.
kappa_mean_net_late = 10 * ed_net_late['kappa'].mean()
kappa_median_net_late = 10 * ed_net_late['kappa'].median()
kappa_std_net_late = 10 * ed_net_late['kappa'].std()
# Compute entry cost as share of winning bid.
kappa_relative_net_late = ed_net_late['kappa'] / ed_net_late['bid_win']
kappa_rel_mean_net_late = 10 * kappa_relative_net_late.mean()
kappa_rel_median_net_late = 10 * kappa_relative_net_late.median()
kappa_rel_std_net_late = 10 * kappa_relative_net_late.std()

# Write results for efficiency comparison to LaTeX table.
with open(project_paths_join('OUT_TABLES', 'entryresults.tex'), 'w') as tex_file:
    # Set a different types of table row with placeholders.
    table_row_header = ' & {val1} & {val2} \\tabularnewline \n'
    table_row_subheader = ' {val1} & & \\tabularnewline \n'
    table_row_stat = ' {stat} & {val1:.4f} & {val2:.4f} \\tabularnewline\n'

    # Top of table.
    tex_file.write('\\begin{tabular}{l|cc}\n\\toprule\n')
    # Header row with names.
    tex_file.write(table_row_header.format(val1='Gross auctions',val2='Net auctions'))
    tex_file.write('\\midrule \n')
    tex_file.write(table_row_subheader.format(val1='\\textbf{Full sample (1995-2011)}'))

    # Write entry costs to table.
    tex_file.write(table_row_stat.format(stat='Mean (in million EUR)',
                                         val1=kappa_mean_gross, val2=kappa_mean_net))
    tex_file.write(table_row_stat.format(stat='Median (in million EUR)',
                                         val1=kappa_median_gross, val2=kappa_median_net))
    tex_file.write(table_row_stat.format(stat='SD (in million EUR)',
                                         val1=kappa_std_gross, val2=kappa_std_net))
    tex_file.write('\\midrule  \n')
    tex_file.write(table_row_subheader.format(val1='\\textbf{Early sample (1995-2004)}'))
    tex_file.write(table_row_stat.format(stat='Mean (in million EUR)',
                                         val1=kappa_mean_gross_early, val2=kappa_mean_net_early))
    tex_file.write(table_row_stat.format(stat='Median (in million EUR)',
                                         val1=kappa_median_gross_early, val2=kappa_median_net_early))
    tex_file.write(table_row_stat.format(stat='SD (in million EUR)',
                                         val1=kappa_std_gross_early, val2=kappa_std_net_early))
    tex_file.write('\\midrule  \n')
    tex_file.write(table_row_subheader.format(val1='\\textbf{Late sample (2005-2011)}'))
    tex_file.write(table_row_stat.format(stat='Mean (in million EUR)',
                                         val1=kappa_mean_gross_late, val2=kappa_mean_net_late))
    tex_file.write(table_row_stat.format(stat='Median (in million EUR)',
                                         val1=kappa_median_gross_late, val2=kappa_median_net_late))
    tex_file.write(table_row_stat.format(stat='SD (in million EUR)',
                                         val1=kappa_std_gross_late, val2=kappa_std_net_late))
    tex_file.write('\\midrule  \n')

    # Add notes to table.
    tex_file.write('\\multicolumn{3}{p{11.75cm}}{\\footnotesize{\\textit{Notes: The table displays summary statistics of the distribution of the estimated entry costs of entrant firms. The upper panel calculates statistics over our whole sample period. The two bottom panel calculates the statistics separately for the early and later years of our sample period.}}}\\tabularnewline\n')
    # Bottom of table.
    tex_file.write('\\bottomrule\n\\end{tabular}\n')


# Write results for efficiency comparison to LaTeX table.
with open(project_paths_join('OUT_TABLES', 'entryresultssmall.tex'), 'w') as tex_file:
    # Set a different types of table row with placeholders.
    table_row_header = ' & {val1} & {val2} \\tabularnewline \n'
    table_row_subheader = ' {val1} & & \\tabularnewline \n'

    table_row_stat = ' {stat} & {val1:.4f} & {val2:.4f} \\tabularnewline\n'

    # Top of table.
    tex_file.write('\\begin{tabular}{l|cc}\n\\toprule\n')
    # Header row with names.
    tex_file.write(table_row_header.format(val1='Gross auctions',val2='Net auctions'))
    tex_file.write('\\midrule \n')
    
    # Write entry costs to table.
    tex_file.write(table_row_stat.format(stat='Median (in million EUR)',
                                         val1=kappa_median_gross, val2=kappa_median_net))
    tex_file.write(table_row_stat.format(stat='SD (in million EUR)',
                                         val1=kappa_std_gross, val2=kappa_std_net))
    tex_file.write('\\midrule  \n')
    
    # Add notes to table.
    tex_file.write('\\multicolumn{3}{p{11.75cm}}{\\footnotesize{\\textit{Notes: The table displays summary statistics of the distribution of the estimated entry costs of entrant firms in the gross and net auction sample, respectively.}}}\\tabularnewline\n')
    # Bottom of table.
    tex_file.write('\\bottomrule\n\\end{tabular}\n')