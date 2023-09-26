/*
	Import and destring data on county incomes.

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
import excel `"${PATH_IN_DATA}/einkommen.xlsx"', sheet("Tabelle1") firstrow clear
* Keep only relevant variables.
keep Regionalschluessel H-U
* Rename income year columns.
foreach v of varlist H-U {
   local x : variable label `v'
   rename `v' einkommen`x'
}
* Destring variables.
* For kkz.
rename Regionalschluessel kkz
destring kkz, replace
* Save file.
save `"${PATH_OUT_DATA}/einkommen.dta"', replace
log close