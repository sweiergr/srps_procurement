/*
	Import number of potential bidders for both samples from Excel-file and 
	export to Stata-format to facilitate merging.

*/

clear
set more off
set mem 2g
version 13
* Header do-File with path definitions, those end up in global macros.
include project_paths
log using `"${PATH_OUT_DATA}/log/`1'.log"', replace
* Load data for gross sample.
import excel `"${PATH_IN_DATA}/ga_pot_bidders.xlsx"', sheet("Sheet1") firstrow clear
rename auctionid id
* Sort and save data.
sort id
save `"${PATH_OUT_DATA}/n_pot_ga.dta"', replace
* Load data for net sample.
import excel `"${PATH_IN_DATA}/na_pot_bidders.xlsx"', sheet("Sheet1") firstrow clear
rename auctionid id
sort id
save `"${PATH_OUT_DATA}/n_pot_na.dta"', replace
log close
