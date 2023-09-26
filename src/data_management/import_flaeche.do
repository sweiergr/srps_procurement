/*
	Import and destring data on county areas.

*/

clear
set more off
set mem 2g
capture log close
version 13
* Header do-File with path definitions, those end up in global macros.
include project_paths
log using `"${PATH_OUT_DATA}/log/`1'.log"', replace
* Import data set.
import excel `"${PATH_IN_DATA}/flaeche.xlsx"', sheet("flaeche") firstrow
* Destring variables.
* For kkz.
destring kkz, replace
drop Kreis
* For area data.
foreach var of varlist flaeche1996-flaeche2012{
	replace `var'="" if `var'=="-"
	replace `var'="" if `var'=="..."
	destring `var', replace
} 
* Save file.
save `"${PATH_OUT_DATA}/flaeche.dta"', replace
log close