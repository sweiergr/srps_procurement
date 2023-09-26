/*
    Descriptive statistics about our data. 
	This is just for exploring descriptive statistics in Stata.
	The statistics reported in the paper are generated by descriptive_statistics.R

*/

clear
capture log close
set more off
set mem 2g
version 13

* Header do-File with path definitions, those end up in global macros.
include project_paths
log using `"${PATH_OUT_ANALYSIS}/log/`1'.log"', replace
di in red("Based on gross and net sample jointly:")
* Load main data set.
use `"${PATH_OUT_DATA}/ga_export.dta"', clear
gen gross = 1 
gen net = 0
append using `"${PATH_OUT_DATA}/na_export.dta"'
replace gross = 0 if gross==.
replace net = 1 if net==.
* Generate time trends.
gen trend = (year - 1996)
gen trend_sq = trend^2
gen trend_net = trend*net
gen trend_net_sq = trend_net^2
* Generate some log variables.
gen frequency_log = log(frequency)
gen n_bidders_log = log(n_bidders)

* Label variables for tables.
label variable net "Net auction"
label variable trend "Time trend"
label variable trend_sq "Time Trend$^2$"
label variable trend_net "Time trend (net)"
label variable trend_net_sq "Time Trend$^2$ (net)"
label variable bid_win "Winning bid" 
label variable EURproNKM "Access charges"
label variable n_bidders "No. of bidders"
label variable zkm_line_prop "Train-km"
label variable laufzeit "Contract duration"
label variable used_vehicles "Used vehicles" 
label variable frequency_log "Frequency (log)"
label variable n_bidders_log "No. of bidders (log)"
* Save data set for use in reduced form regressions.
save `"${PATH_OUT_DATA}/rf_regdata.dta"', replace

* Label values of net auction.
label define net_label 0 "Gross Auctions" 1 "Net Auctions"
label val net net_label

* Print descriptive statistics on gross and net auctions.
local reg_list "bid_win n_bidders db_win laufzeit EURproNKM zkm_line_prop used_vehicles"
bysort net: sum `reg_list'
* Use outreg2 command to format table of descriptive statistics.
outreg2 using `"${PATH_OUT_TABLES}/descriptivesfull.tex"', label tex(frag) addnote("Notes: The table presents summary statistics of", " key variables for our full sample.") replace sum(log) keep(bid_win n_bidders laufzeit zkm_line_prop used_vehicles)
bysort net: outreg2 using `"${PATH_OUT_TABLES}/descriptivessplit.tex"', replace tex(frag)  sum(log) addnote("Note: Summary statistics of key variables \\linebreak split by auction mode: gross vs. net.") keep(bid_win n_bidders laufzeit zkm_line_prop used_vehicles) label
tabstat bid_win n_bidders, by(net) statistics(mean count min max sd)
* Create table for gross and net separately.
preserve
keep if net==0
outreg2 using `"${PATH_OUT_TABLES}/descriptivesgross.tex"', label tex(frag) replace sum(log) addnote("Note: Summary statistics of key variables for sample of auctions without revenue risk (gross auctions).") keep(bid_win n_bidders laufzeit zkm_line_prop used_vehicles)
restore
preserve
keep if net==1
outreg2 using `"${PATH_OUT_TABLES}/descriptivesnet.tex"', label tex(frag) replace sum(log) addnote("Note: Summary statistics of key variables for sample of auctions with revenue risk (net auctions).")  keep(bid_win n_bidders laufzeit zkm_line_prop used_vehicles)
restore

* Create dummy variable for first and last 10 years of sample.
gen late_period = 1 if year>2006
replace late_period = 0 if late_period==.
* See that relative winning of DB decreases in net over time (?)
bysort late_period: tab db_win net

* Do t-tests on equality of means of track characteristics regressors.
local regtest_list "laufzeit EURproNKM zkm_line_prop used_vehicles diesel"
foreach var in `regtest_list'{
	di in red( "Tested regressor: `var' ")
	ttest `var', une by(net)
	bysort late_period: ttest `var', une by(net)
}

estpost ttest bid_win zkm used_*, by(net) 
esttab using `"${PATH_OUT_TABLES}/comparegrossnetbids.tex"', ///
cells("mu_1(fmt(2) label(net)) N_1 (fmt(%9.0g) label(N)) mu_2(fmt(2) label(gross))  N_2(fmt(%9.0g) label(N)) b (fmt(2) label(Difference))  p (fmt(2) label(p-value)) ") ///
nonumber label tex replace

***************************************************
* Overview table
***************************************************
estpost ttest laufzeit EURproNKM zkm_line_prop used_vehicles, une by(net) 
esttab using `"${PATH_OUT_TABLES}/comparegrossnetcharacteristics.tex"', addnote("Notes: This table compares the means of key track characteristics across" " gross and net sample. The last columns corresponds to the p-value for testing" "equality of the mean across gross and net sample.")  ///
cells("mu_1(fmt(2) label(net)) N_1 (fmt(%9.0g) label(N)) mu_2(fmt(2) label(gross))  N_2(fmt(%9.0g) label(N)) b (fmt(2) label(Difference))  p (fmt(2) label(p-value)) ") ///
nonumber label tex replace