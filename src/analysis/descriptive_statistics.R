# Create table of descriptive statistics, reported in Section 3 of the paper.
# Load required libraries.
library(tidyverse)
library(haven)
library(forcats)
library(qwraps2)
# Clear workspace. 
rm(list=ls())
# Load project path definitions.
source("project_paths.R")

auction_mode_label <- c("Gross", "Net")
gross_data <- read_dta(str_c(PATH_OUT_DATA,"/","ga_export.dta")) %>%
    mutate(Mode = "Gross")
# We drop one outlier in the gross auction sample, that has a winning bid that is more than twice of the second largest bid.
gross_outlier <- filter(gross_data,bid_win>55)
gross_data <- filter(gross_data,bid_win<55)
net_data <- read_dta(str_c(PATH_OUT_DATA,"/","na_export.dta")) %>%
    mutate(Mode = "Net")
data_combined <- bind_rows(gross_data,net_data)
# Transform auction mode to factor variable.
data_combined$Mode <- factor(data_combined$Mode, levels = auction_mode_label)
data_group <- group_by(data_combined, Mode)
data_combined <- mutate(data_combined, laufzeit = laufzeit * 10)
data_combined <- mutate(data_combined, EURproNKM = EURproNKM * 10)

# Basic descriptive statistics.
dstats <- 
    list("Winning bid (10 Mio.~EUR)" = 
             list("Mean" = ~mean(bid_win),
                  "SD" = ~ sd(bid_win),
                  "Min" = ~ min(bid_win), 
                  "Max" = ~ max(bid_win)),
         "No.~bidders" = 
             list("Mean" = ~mean(n_bidders),
                  "SD" = ~ sd(n_bidders),
                  "Min" = ~ min(n_bidders), 
                  "Max" = ~ max(n_bidders)),
         "Access charges (EUR)" = 
             list("Mean" = ~mean(EURproNKM),
                  "SD" = ~ sd(EURproNKM),
                  "Min" = ~ min(EURproNKM), 
                  "Max" = ~ max(EURproNKM)),
         "Volume (Mio.~train-km)" = 
             list("Mean" = ~mean(zkm_line_prop),
                  "SD" = ~ sd(zkm_line_prop),
                  "Min" = ~ min(zkm_line_prop), 
                  "Max" = ~ max(zkm_line_prop)),
         "Train frequency (per day)" = 
             list("Mean" = ~mean(frequency),
                  "SD" = ~ sd(frequency),
                  "Min" = ~ min(frequency), 
                  "Max" = ~ max(frequency)),
         "Size of network (km)" = 
             list("Mean" = ~mean(nkm),
                  "SD" = ~ sd(nkm),
                  "Min" = ~ min(nkm), 
                  "Max" = ~ max(nkm)),
         "Duration (Years) " = 
             list("Mean" = ~mean(laufzeit),
                  "SD" = ~ sd(laufzeit),
                  "Min" = ~ min(laufzeit), 
                  "Max" = ~ max(laufzeit)),
         "Used vehicles (Dummy)" = 
             list("Mean" = ~mean(used_vehicles),
                  "SD" = ~ sd(used_vehicles),
                  "Min" = ~ min(used_vehicles), 
                  "Max" = ~ max(used_vehicles))
         ) 
n_obs_gross = nrow(gross_data)
n_obs_net = nrow(net_data)

test_table <- summary_table(group_by(data_combined, Mode),dstats)
var_names <- names(dstats)
stat_names <- names(dstats[[1]])
ds_line <- " %s & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\\\ "
ds_line_title <- "  & %s & %s & %s & %s & %s & %s & %s & %s \\\\"
fileConn<-str_c(PATH_OUT_TABLES,"/","dssplitnewNoOutlier.tex")
write("\\begin{tabular}{l|cccc|cccc}", fileConn)
write("\\toprule", fileConn,append=T)
write(sprintf(" & \\textbf{Gross} & (N=%i) & & & \\textbf{Net} & (N=%i) & & \\\\",
              n_obs_gross, n_obs_net),fileConn,append=T)
write(sprintf(ds_line_title, stat_names[1], stat_names[2],
              stat_names[3], stat_names[4],
              stat_names[1], stat_names[2],
              stat_names[3], stat_names[4]),
      fileConn,append=T)
write("\\midrule",fileConn,append=T)

# Loop over variables.
for (var_i in seq_along(dstats)){
    write(sprintf(ds_line, var_names[var_i], 
                  as.double(test_table[(var_i-1)*4+1,1]), as.double(test_table[(var_i-1)*4+2,1]),
                  as.double(test_table[(var_i-1)*4+3,1]), as.double(test_table[(var_i-1)*4+4,1]),
                  as.double(test_table[(var_i-1)*4+1,2]), as.double(test_table[(var_i-1)*4+2,2]),
                  as.double(test_table[(var_i-1)*4+3,2]), as.double(test_table[(var_i-1)*4+4,2])
                  ),
          fileConn,append=T)
}
write("\\midrule",fileConn,append=T)
write("\\multicolumn{9}{p{14.5cm}}{\\footnotesize{\\textit{Notes: 
      This table compares descriptive statistics of the most important auction
      characteristics across different auction modes (gross vs. net). \\emph{Volume}
      captures the size of the contract in million train-km, i.e., the total number 
      of km that one train would have to drive to fulfill the contract, \\emph{Train frequency}
      indicates how often a train has to serve the track per day, \\emph{Size of network} 
      describes the size of the physical track network to be covered (in km), \\emph{Duration}
      is the length of the contract, \\emph{Used vehicles} indicates whether the contract requires new vehicles
      to be used on the track with 1 indicating that used vehicles are permitted.
     \\emph{Access charges} denotes the regulated price (in EUR) that a firm has to 
        pay every time a train operates on a one km long stretch of a specific track. }}}\\\\",
      fileConn,append=T)
write("\\bottomrule",fileConn,append=T)
write("\\end{tabular}",fileConn,append=T)

## Table 2: Formal t-tests and plot differences in track characteristics.
tt_bidwin <- t.test(bid_win~Mode,data=data_combined)
tt_nbidders <- t.test(n_bidders~Mode,data=data_combined)
# t-test for comparison of means.
tt_EURproNKM <- t.test(EURproNKM~Mode, data = data_combined)
tt_zkm <- t.test(zkm_line_prop~Mode, data = data_combined)
tt_laufzeit <- t.test(laufzeit~Mode, data = data_combined)
tt_used_vehicles <- t.test(used_vehicles~Mode, data = data_combined)
# Add new variables that AEJ/R1 requested.
tt_frequency <- t.test(frequency~Mode, data = data_combined)
tt_nkm <- t.test(nkm~Mode, data = data_combined)

# Collect p-values in one list.
p_values = list(tt_EURproNKM$p.value, tt_zkm$p.value,
                tt_frequency$p.value, tt_nkm$p.value,
                tt_laufzeit$p.value,tt_used_vehicles$p.value)

p_values_comb = list(tt_bidwin$p.value, tt_nbidders$p.value, tt_EURproNKM$p.value, tt_zkm$p.value,
                tt_frequency$p.value, tt_nkm$p.value,
                tt_laufzeit$p.value,tt_used_vehicles$p.value)

mean_diff = list(tt_EURproNKM$estimate[1]-tt_EURproNKM$estimate[2],
                 tt_zkm$estimate[1]-tt_zkm$estimate[2],
                 tt_frequency$estimate[1]-tt_frequency$estimate[2],
                 tt_nkm$estimate[1]-tt_nkm$estimate[2],
                 tt_laufzeit$estimate[1]-tt_laufzeit$estimate[2],
                 tt_used_vehicles$estimate[1]-tt_used_vehicles$estimate[2])
mean_gross = list(tt_EURproNKM$estimate[1],
                  tt_zkm$estimate[1],
                  tt_frequency$estimate[1],
                  tt_nkm$estimate[1],
                  tt_laufzeit$estimate[1],
                  tt_used_vehicles$estimate[1])
mean_net = list(tt_EURproNKM$estimate[2],
                tt_zkm$estimate[2],
                tt_frequency$estimate[2],
                tt_nkm$estimate[2],
                tt_laufzeit$estimate[2],
                tt_used_vehicles$estimate[2])
# Construct LaTeX table code.
var_names_2 = list("Access charges (EUR per net-km)",
                   "Volume (Mio.~train-km)",
                   "Train frequency (per day)",
                   "Size of network (km)",
                   "Duration (Years)",
                   "Used vehicles (Dummy)")
tt_line <- " %s & %.4f & %.4f & %.4f & %.4f \\\\ "
tt_line_title <- "  & %s & %s & %s & %s  \\\\"
fileConn<-str_c(PATH_OUT_TABLES,"/","ttesttrackcharNoOutlier.tex")
write("\\begin{tabular}{l|cc|cc}", fileConn)
write("\\toprule", fileConn,append=T)
write(sprintf(" & \\textbf{Gross} & \\textbf{Net} &  & \\\\"),fileConn,append=T)
write(sprintf(" & (N=%i)&  (N=%i) & & \\\\",
              n_obs_gross, n_obs_net),fileConn,append=T)
write(sprintf(tt_line_title, "Mean", "Mean", "Difference","p-value"),
      fileConn,append=T)
write("\\midrule",fileConn,append=T)

# Loop over variables.
for (var_i in seq_along(var_names_2)){
  write(sprintf(tt_line, var_names_2[var_i], 
                as.double(mean_gross[var_i]), as.double(mean_net[var_i]),
                as.double(mean_diff[var_i]), as.double(p_values[var_i])),
  fileConn,append=T)
}

write("\\midrule",fileConn,append=T)
write("\\multicolumn{5}{p{13.5cm}}{\\footnotesize{\\textit{Notes: This table summarizes the results from 
      testing the equality of the means of the most important track characteristics across different auction modes (gross vs. net).
      Variable definitions are as in Table 1.
      }}}\\\\",
      fileConn,append=T)
write("\\bottomrule",fileConn,append=T)
write("\\end{tabular}",fileConn,append=T)


################################################################################
# COMBINED TABLE 1 AND 2 FOR IJIO REVISION/R1.
ds_line_comb <- " %s & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\\\ "
ds_line_title_comb <- "  & %s & %s & %s & %s & %s & %s & %s & %s & %s \\\\"
fileConn<-str_c(PATH_OUT_TABLES,"/","dssplitnewNoOutlierComb.tex")
write("\\begin{tabular}{l|cccc|cccc|c}", fileConn)
write("\\toprule", fileConn,append=T)
write(sprintf(" & \\textbf{Gross} & (N=%i) & & & \\textbf{Net} & (N=%i) & & & \\textbf{p-value} \\\\",
              n_obs_gross, n_obs_net),fileConn,append=T)
write(sprintf(ds_line_title_comb, stat_names[1], stat_names[2],
              stat_names[3], stat_names[4],
              stat_names[1], stat_names[2],
              stat_names[3], stat_names[4], ''),
      fileConn,append=T)
write("\\midrule",fileConn,append=T)

# Loop over variables.
for (var_i in seq_along(dstats)){
    write(sprintf(ds_line_comb, var_names[var_i], 
                  as.double(test_table[(var_i-1)*4+1,1]), as.double(test_table[(var_i-1)*4+2,1]),
                  as.double(test_table[(var_i-1)*4+3,1]), as.double(test_table[(var_i-1)*4+4,1]),
                  as.double(test_table[(var_i-1)*4+1,2]), as.double(test_table[(var_i-1)*4+2,2]),
                  as.double(test_table[(var_i-1)*4+3,2]), as.double(test_table[(var_i-1)*4+4,2]), 
                  as.double(p_values_comb[var_i])
    ),
    fileConn,append=T)
    if (var_i==2)
    {
        write("\\midrule",fileConn,append=T)
    }
}
write("\\midrule",fileConn,append=T)
write("\\multicolumn{10}{p{16cm}}{\\footnotesize{\\textit{Notes: 
      This table compares descriptive statistics of the most important auction
      characteristics across different auction modes (gross vs. net). \\emph{Volume}
      captures the size of the contract in million train-km, i.e., the total number 
      of km that one train would have to drive to fulfill the contract, \\emph{Train frequency}
      indicates how often a train has to serve the track per day, \\emph{Size of network} 
      describes the size of the physical track network to be covered (in km), \\emph{Duration}
      is the length of the contract, \\emph{Used vehicles} indicates whether the contract requires new vehicles
      to be used on the track with 1 indicating that used vehicles are permitted.
     \\emph{Access charges} denotes the regulated price (in EUR) that a firm has to 
      pay every time a train operates on a one km long stretch of a specific track. The last column displays the p-values associated with 
      testing the equality of the means of the most important track characteristics across different auction modes (gross vs. net).}}}\\\\",
      fileConn,append=T)

write("\\bottomrule",fileConn,append=T)
write("\\end{tabular}",fileConn,append=T)

################################################################################
# Repeat above table but including one outlier observation in gross contract.
auction_mode_label <- c("Gross", "Net")
gross_data <- read_dta(str_c(PATH_OUT_DATA,"/","ga_export.dta")) %>%
  mutate(Mode = "Gross")
# Here, we don't drop the outlier.
gross_outlier <- filter(gross_data,bid_win>55)

net_data <- read_dta(str_c(PATH_OUT_DATA,"/","na_export.dta")) %>%
  mutate(Mode = "Net")
data_combined <- bind_rows(gross_data,net_data)
data_combined$Mode <- factor(data_combined$Mode, levels = auction_mode_label)
data_group <- group_by(data_combined, Mode)
data_combined <- mutate(data_combined, laufzeit = laufzeit * 10)
data_combined <- mutate(data_combined, EURproNKM = EURproNKM * 10)
dstats <- 
  list("Winning bid (10 Mio.~EUR)" = 
         list("Mean" = ~mean(bid_win),
              "SD" = ~ sd(bid_win),
              "Min" = ~ min(bid_win), 
              "Max" = ~ max(bid_win)),
       "No.~bidders" = 
         list("Mean" = ~mean(n_bidders),
              "SD" = ~ sd(n_bidders),
              "Min" = ~ min(n_bidders), 
              "Max" = ~ max(n_bidders)),
       "Access charges (EUR)" = 
         list("Mean" = ~mean(EURproNKM),
              "SD" = ~ sd(EURproNKM),
              "Min" = ~ min(EURproNKM), 
              "Max" = ~ max(EURproNKM)),
       "Volume (Mio.~train-km)" = 
         list("Mean" = ~mean(zkm_line_prop),
              "SD" = ~ sd(zkm_line_prop),
              "Min" = ~ min(zkm_line_prop), 
              "Max" = ~ max(zkm_line_prop)),
       "Train frequency (per day)" = 
         list("Mean" = ~mean(frequency),
              "SD" = ~ sd(frequency),
              "Min" = ~ min(frequency), 
              "Max" = ~ max(frequency)),
       "Size of network (km)" = 
         list("Mean" = ~mean(nkm),
              "SD" = ~ sd(nkm),
              "Min" = ~ min(nkm), 
              "Max" = ~ max(nkm)),
       "Duration (Years) " = 
         list("Mean" = ~mean(laufzeit),
              "SD" = ~ sd(laufzeit),
              "Min" = ~ min(laufzeit), 
              "Max" = ~ max(laufzeit)),
       "Used vehicles (Dummy)" = 
         list("Mean" = ~mean(used_vehicles),
              "SD" = ~ sd(used_vehicles),
              "Min" = ~ min(used_vehicles), 
              "Max" = ~ max(used_vehicles))
  ) 
n_obs_gross = nrow(gross_data)
n_obs_net = nrow(net_data)
test_table <- summary_table(group_by(data_combined, Mode),dstats)
var_names <- names(dstats)
stat_names <- names(dstats[[1]])
ds_line <- " %s & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\\\ "
ds_line_title <- "  & %s & %s & %s & %s & %s & %s & %s & %s \\\\"
fileConn<-str_c(PATH_OUT_TABLES,"/","dssplitnew.tex")
write("\\begin{tabular}{l|cccc|cccc}", fileConn)
write("\\toprule", fileConn,append=T)
write(sprintf(" & \\textbf{Gross} & (N=%i) & & & \\textbf{Net} & (N=%i) & & \\\\",
              n_obs_gross, n_obs_net),fileConn,append=T)
write(sprintf(ds_line_title, stat_names[1], stat_names[2],
              stat_names[3], stat_names[4],
              stat_names[1], stat_names[2],
              stat_names[3], stat_names[4]),
      fileConn,append=T)
write("\\midrule",fileConn,append=T)

# Loop over variables.
for (var_i in seq_along(dstats)){
  write(sprintf(ds_line, var_names[var_i], 
                as.double(test_table[(var_i-1)*4+1,1]), as.double(test_table[(var_i-1)*4+2,1]),
                as.double(test_table[(var_i-1)*4+3,1]), as.double(test_table[(var_i-1)*4+4,1]),
                as.double(test_table[(var_i-1)*4+1,2]), as.double(test_table[(var_i-1)*4+2,2]),
                as.double(test_table[(var_i-1)*4+3,2]), as.double(test_table[(var_i-1)*4+4,2])
  ),
  fileConn,append=T)
}
write("\\midrule",fileConn,append=T)
write("\\multicolumn{9}{p{14.5cm}}{\\footnotesize{\\textit{Notes: 
      This table compares descriptive statistics of the most important auction
      characteristics across different auction modes (gross vs. net). \\emph{Volume}
      captures the size of the contract in million train-km, i.e., the total number 
      of km that one train would have to drive to fulfill the contract, \\emph{Train frequency}
      indicates how often a train has to serve the track per day, \\emph{Size of network} 
      describes the size of the physical track network to be covered (in km), \\emph{Duration}
      is the length of the contract, \\emph{Used vehicles} indicates whether the contract requires new vehicles
      to be used on the track with 1 indicating that used vehicles are permitted.
     \\emph{Access charges} denotes the regulated price (in EUR) that a firm has to 
        pay every time a train operates on a one km long stretch of a specific track. }}}\\\\",
      fileConn,append=T)

write("\\bottomrule",fileConn,append=T)
write("\\end{tabular}",fileConn,append=T)

# Table 2: formal t-tests and plot differences in track characteristics.
# t-test for comparison of means.
tt_EURproNKM <- t.test(EURproNKM~Mode, data = data_combined)
tt_zkm <- t.test(zkm_line_prop~Mode, data = data_combined)
tt_laufzeit <- t.test(laufzeit~Mode, data = data_combined)
tt_used_vehicles <- t.test(used_vehicles~Mode, data = data_combined)
# Add new variables that AEJ/R1 requested.
tt_frequency <- t.test(frequency~Mode, data = data_combined)
tt_nkm <- t.test(nkm~Mode, data = data_combined)
# Collect p-values in one list.
p_values = list(tt_EURproNKM$p.value, tt_zkm$p.value,
                tt_frequency$p.value, tt_nkm$p.value,
                tt_laufzeit$p.value,tt_used_vehicles$p.value)
mean_diff = list(tt_EURproNKM$estimate[1]-tt_EURproNKM$estimate[2],
                 tt_zkm$estimate[1]-tt_zkm$estimate[2],
                 tt_frequency$estimate[1]-tt_frequency$estimate[2],
                 tt_nkm$estimate[1]-tt_nkm$estimate[2],
                 tt_laufzeit$estimate[1]-tt_laufzeit$estimate[2],
                 tt_used_vehicles$estimate[1]-tt_used_vehicles$estimate[2])
mean_gross = list(tt_EURproNKM$estimate[1],
                  tt_zkm$estimate[1],
                  tt_frequency$estimate[1],
                  tt_nkm$estimate[1],
                  tt_laufzeit$estimate[1],
                  tt_used_vehicles$estimate[1])
mean_net = list(tt_EURproNKM$estimate[2],
                tt_zkm$estimate[2],
                tt_frequency$estimate[2],
                tt_nkm$estimate[2],
                tt_laufzeit$estimate[2],
                tt_used_vehicles$estimate[2])
# Construct LaTeX table code.
var_names_2 = list("Access charges (EUR per net-km)",
                   "Volume (Mio.~train-km)",
                   "Train frequency (per day)",
                   "Size of network (km)",
                   "Duration (Years)",
                   "Used vehicles (Dummy)")
tt_line <- " %s & %.4f & %.4f & %.4f & %.4f \\\\ "
tt_line_title <- "  & %s & %s & %s & %s  \\\\"
fileConn<-str_c(PATH_OUT_TABLES,"/","ttesttrackchar.tex")
write("\\begin{tabular}{l|cc|cc}", fileConn)
write("\\toprule", fileConn,append=T)
write(sprintf(" & \\textbf{Gross} & \\textbf{Net} &  & \\\\"),fileConn,append=T)
write(sprintf(" & (N=%i)&  (N=%i) & & \\\\",
              n_obs_gross, n_obs_net),fileConn,append=T)
write(sprintf(tt_line_title, "Mean", "Mean", "Difference","p-value"),
      fileConn,append=T)
write("\\midrule",fileConn,append=T)
# Loop over variables.
for (var_i in seq_along(var_names_2)){
  write(sprintf(tt_line, var_names_2[var_i], 
                as.double(mean_gross[var_i]), as.double(mean_net[var_i]),
                as.double(mean_diff[var_i]), as.double(p_values[var_i])),
        fileConn,append=T)
}
write("\\midrule",fileConn,append=T)
write("\\multicolumn{5}{p{13.5cm}}{\\footnotesize{\\textit{Notes: This table summarizes the results from 
      testing the equality of the means of the most important track characteristics across different auction modes (gross vs. net).
      Variable definitions are as in Table 1.
      }}}\\\\",
      fileConn,append=T)
write("\\bottomrule",fileConn,append=T)
write("\\end{tabular}",fileConn,append=T)