/*
	Clean, i.p., destring information on population.

*/

clear
set more off
set mem 2g
capture log close
version 13
* Header do-File with path definitions, those end up in global macros.
include project_paths
log using `"${PATH_OUT_DATA}/log/`1'.log"', replace

* Load main data set.
use `"${PATH_IN_DATA}/BevoelkerungKKZ.dta"', clear
foreach var of varlist y1996-y2012{
	replace `var'="" if `var'=="-"
	destring `var', replace
} 
* Save file.
save `"${PATH_OUT_DATA}/bevoelkerungkkz.dta"', replace
log close