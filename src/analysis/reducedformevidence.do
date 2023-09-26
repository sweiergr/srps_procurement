/*
    Provide reduced form evidence of gross and net tracks being similar, but bids systematically different.

*/

clear
capture cls
capture log close
set more off
set mem 1g
version 13

* Header do-File with path definitions, those end up in global macros.
include project_paths
log using `"${PATH_OUT_ANALYSIS}/log/`1'.log"', replace
* Reduced-form evidence based on joint sample of gross and net auctions.
clear
di in red("Reduced-form evidence based on gross and net sample jointly:")
use `"${PATH_OUT_DATA}/rf_regdata.dta"', clear

* Analyze winning bids.
gen n_net = n_bidders * net
label var n_net "No. bidders-net"
gen n_gross = n_bidders * (1-net)
label var n_gross "No. bidders-gross"
* Minor updates to labels of variables.
label var laufzeit "Contract duration"
label var zkm_line_prop "Contract volume"
reg bid_win trend trend_sq n_gross n_net net nkm laufzeit frequency_log zkm_line_prop EURproNKM used_vehicles diesel, robust
eststo winning_bid

** Analyze number of bidders.
reg n_bidders trend trend_sq net laufzeit frequency_log zkm_line_prop EURproNKM used_vehicles nkm, robust
eststo number_bidders

predict n_bidders_pred_ols
gen n_bidders_pred_ols_gross = n_bidders_pred_ols - 1.805897 if net==1
gen n_bidders_pred_ols_net = n_bidders_pred_ols + 1.805897 if gross==1
replace n_bidders_pred_ols_gross = n_bidders_pred_ols if net==0
replace n_bidders_pred_ols_net = n_bidders_pred_ols if net==1

poisson n_bidders trend trend_sq net laufzeit frequency_log zkm_line_prop EURproNKM used_vehicles nkm , robust
eststo number_bidders_poisson
esttab winning_bid number_bidders number_bidders_poisson using `"${PATH_OUT_TABLES}/rfbidsn.tex"', ///
	se noobs  keep(n_net n_gross net laufzeit frequency_log zkm_line_prop EURproNKM used_vehicles) varlabels(_cons Constant diesel Diesel) ///
	cells(b(star fmt(4)) se(par fmt(4))) ///
	ar2 tex label eqlabels(none)  ///
	collabels(none) ///
	order(net n_gross n_net EURproNKM zkm_line_prop laufzeit used_vehicles frequency_log) ///
	star(* 0.10 ** 0.05 *** 0.01)     ///
	replace nonotes ///
	addnotes("\textit{Notes:} Heteroskedasticity-robust standard errors in parentheses. Models (1) and (2)" ///
			 "are estimated by OLS. Column (3) is a negative binomial count data regression.  " ///
			 "\emph{No.~bidders-gross} and \emph{No.~bidders-net} denote the number of bidders in the auction" ///
			 "interacted with dummies for gross and net auctions, respectively."  ///
			 "\emph{Net auction} is a dummy for net auctions. \emph{Frequency (log)} is the logged average" ///
			 "number of times the train has to operate on the line per day. All other variable" ///  
			 " definitions are as in Table 1. Number of observations: 157." ///
			 "* p < 0.1, ** p < 0.05, *** p < 0.01." ///
			 )

* Check whether reduced form model predicts a meaningful number of bidders.
predict n_bidders_pred_count

** Analyze probability of DB winning.
gen nb_trunc = min(n_bidders, 5)
gen n_net_trunc = min(n_net,5)
gen n_gross_trunc = min(n_gross,5)
gen n_net_log = net * log(n_bidders)
gen n_gross_log = gross * log(n_bidders)

* Without controlling for number of bidders.
* This specification is not used in paper anymore.
logit db_win nkm used_vehicles frequency_log net zkm_line_prop laufzeit EURproNKM i.year trend trend_net, robust
eststo inc_win_base
* Controlling for observed number of bidders.
logit db_win nkm n_net n_gross used_vehicles frequency_log net zkm_line_prop laufzeit EURproNKM  trend  trend_net i.year
eststo inc_win_nbidders

* Compute average of our statistics of potential number of bidders.
gen n_pot = (1/3) * (n_pot1 + n_pot2 + n_pot3)
gen n_pot_net = n_pot * net
gen n_pot_gross = n_pot * (1-net)
label var n_pot_gross "Pot. bidders-gross"
label var n_pot_net "Pot. bidders-net"
label var n_pot "No. of pot. bidders"
* With potential number of bidders as control (averaged over our three different measures).
* All specs are roughly similar. This is my currently most preferred spec.
logit db_win nkm n_pot_gross n_pot_net used_vehicles frequency_log net zkm_line_prop laufzeit EURproNKM trend trend_net i.year
eststo inc_win_npotmean

logit db_win nkm n_net  used_vehicles frequency_log net zkm_line_prop c.laufzeit trend  EURproNKM trend_net, robust 
* Redo with predicted number of bidders from count data model.
gen n_pred_net = n_bidders_pred_count * net
gen n_pred_gross = n_bidders_pred_count * (1-net)
label var n_pred_gross "Pred. bidders-gross"
label var n_pred_net "Pred. bidders-net"
logit db_win nkm n_bidders_pred_count used_vehicles frequency_log net zkm_line_prop laufzeit EURproNKM i.year trend_net, robust
eststo inc_win_npred

** Analyze choice of net auction.
logit net zkm_line_prop nkm frequency_log laufzeit EURproNKM used_vehicles, robust
eststo net_choice

label var nkm "Network size"
label var n_bidders_pred_count "Pred. no. of bidders"
* This is the regression table for the current paper draft.
* Compared to the initial table, we now present two more specifications that control for the number of bidders.
esttab inc_win_base inc_win_nbidders inc_win_npotmean net_choice using `"${PATH_OUT_TABLES}/rfdbwinauctionmode.tex"', ///
	se noobs ///
	cells(b(star fmt(4)) se(par fmt(4))) ///
	keep(net laufzeit frequency_log zkm_line_prop EURproNKM used_vehicles frequency_log net n_net n_gross n_pot_gross n_pot_net zkm_line_prop laufzeit EURproNKM used_vehicles nkm) varlabels(_cons Constant) ///	cells(b(star fmt(4)) se(par fmt(4))) ///
	tex label interaction(" X ") eqlabels(none) collabels(none) ///
	star(* 0.10 ** 0.05 *** 0.01)     ///
	order(net EURproNKM zkm_line_prop laufzeit used_vehicles frequency_log) ///
	replace nonotes ///
	addnotes("\textit{Notes:} Heteroskedasticity-robust standard errors in parentheses." ///
		     "All models are estimated using binary logit regressions. \emph{DB wins}" ///
		     "denotes a dummy indicating whether the incumbent won the auction." ///
		     "\emph{Net auction} is 1 (0) if the auction is a net (gross) auction." ///
		     "All other variable definitions are as in Table 2." ///
			 "Number of observations: 157. * p < 0.1, ** p < 0.05, *** p < 0.01" ///
			 )


// Format large table with all reduced form results to be presented in appendix.
* Compared to the initial table, we now present two more specifications that control for the number of bidders.
esttab winning_bid number_bidders_poisson inc_win_base inc_win_nbidders inc_win_npotmean net_choice using `"${PATH_OUT_TABLES}/rf_full_appendix.tex"', ///
	se noobs ///
	cells(b(star fmt(4)) se(par fmt(4))) ///
	keep(n_net n_gross net laufzeit frequency_log zkm_line_prop EURproNKM used_vehicles frequency_log net  n_pot_net n_pot_gross zkm_line_prop laufzeit EURproNKM used_vehicles nkm) varlabels(_cons Constant) ///	cells(b(star fmt(4)) se(par fmt(4))) ///
	tex label interaction(" X ") eqlabels(none) collabels(none) ///
	star(* 0.10 ** 0.05 *** 0.01)     ///
	order(net EURproNKM zkm_line_prop laufzeit used_vehicles frequency_log) ///
	replace nonotes ///
	addnotes("\textit{Notes:} Heteroskedasticity-robust standard errors in parentheses. Models (1) is estimated by OLS. Column (2) is a negative binomial" ///
			 "count data regression. Models (3) to (6) are binary logit models. \emph{No.~bidders-gross} and \emph{No.~bidders-net} denote the number of" ///
			 "bidders in the auction interacted with dummies for gross and net auctions, respectively. \emph{Net auction} is a dummy for net auctions." ///
			 "\emph{Frequency (log)} is the logged average number of times the train has to operate on the line per day. \emph{DB wins} denotes a" ///
		     "dummy indicating whether the incumbent won the auction. \emph{Net auction} is 1 (0) if the auction is a net (gross) auction." ///
			 "All other variable definitions are as in Table 1. Number of observations: 156. * p < 0.1, ** p < 0.05, *** p < 0.01" ///
			 )