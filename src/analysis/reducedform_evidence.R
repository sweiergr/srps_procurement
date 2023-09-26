# Updated reduced form evidence, reported in Section 3 of the IJIO paper.
# Load required libraries.
library(tidyverse)
library(haven)
library(forcats)
library(plm)
library(sandwich)
library(stargazer)
library(dummies)
library(apsrtable)
# Clear workspace. 
rm(list=ls())

# Load project path definitions.
source("project_paths.R")

# Run reduced form regressions based on joint sample.
full_sample <- read_dta(str_c(PATH_OUT_DATA,"/","rf_regdata.dta"))
# Create a list of variable labels.
var.labels <- c(n_net = "No. bidders-net", 
                n_gross = "No. bidders-gross")
# Generate some more variables.
reg_data <- full_sample %>%
    mutate(n_net = n_bidders * net,
           n_gross = n_bidders * (1-net),
           year_fac = as.factor(year))

# Run OLS regressions for winning bid and number of bidders.
winning_bid <- lm(data = reg_data,
                  formula= bid_win ~ 
                  trend + trend_sq + n_gross + n_net
                  + net + laufzeit + frequency_log +
                  zkm_line_prop + EURproNKM + used_vehicles +
                  + diesel)
# Compute robust standard errors.
winning_bid.robust.vcov <- vcovHC(winning_bid,type = "HC1")
winning_bid.robust.se <- sqrt(diag(winning_bid.robust.vcov))
# Add new robust standard errors to printed table.
stargazer(winning_bid, type="text", se=list(winning_bid.robust.se))
stargazer(winning_bid, type="text")

# Reduced form regressions for number of bidders.
no_bidders <- lm(data= reg_data,
                formula = n_bidders ~ 
                trend + trend_sq + net + laufzeit
                + frequency_log +
                zkm_line_prop + EURproNKM + used_vehicles +
                 + diesel)
# Compute robust standard errors.
no_bidders.robust.vcov <- vcovHC(no_bidders,type = "HC1")
no_bidders.robust.se <- sqrt(diag(no_bidders.robust.vcov))
# Add new robust standard errors to printed table.
stargazer(no_bidders, type="text", se=list(no_bidders.robust.se))
stargazer(no_bidders, type="text")

# Logit regression of DB winning.
db_wins <- glm(data = reg_data,
    formula = db_win ~
        frequency_log + net + zkm_line_prop + laufzeit + 
        EURproNKM + trend_net + factor(year), family = "binomial"(link="logit"))
# Robust Standard Errors
robust.cov.db_wins <- vcovHC(db_wins,type="HC0")
robust.se.db_wins <- sqrt(diag(robust.cov.db_wins))
stargazer(db_wins,type="text",se=list(robust.se.db_wins))

# Compute predicted probabilities as a sanity check.
reg_data$db_win_pred <- predict(db_wins, data=reg_data, type = "response")

# Logit regression of when net auctions are procured.
net_mode <- glm(data = reg_data,
               formula = net ~
                   zkm_line_prop + nkm + frequency_log +
                   laufzeit + EURproNKM + used_vehicles, family = "binomial"(link="logit"))
# Robust Standard Errors
robust.cov.net_mode <- vcovHC(net_mode,type="HC1")
robust.se.net_mode <- sqrt(diag(robust.cov.net_mode))
stargazer(net_mode,type="text",se=list(robust.se.net_mode))

# Export tables to LaTeX files.
stargazer(db_wins, net_mode,type="text", 
          se=list(robust.se.db_wins,robust.se.net_mode),
          keep=c("zkm_line_prop","nkm","frequency_log","laufzeit","EURproNKM","used_vehicles",
                 "net"),
          covariate.labels = c("Log(frequency)","Net auction","Train-km","Contract duration","Access charges")
          )