/*
	For sample of direct negotiations,
	import track cost data form Excel file to Stata file.
	This part of the sample is NOT used in the current paper.

*/
clear
set more off
set mem 2g
version 13
* Header do-File with path definitions, those end up in global macros.
include project_paths
log using `"${PATH_OUT_DATA}/log/`1'.log"', replace
* Load main data set.
import excel `"${PATH_IN_DATA}/fill_cost_data_negotiations_edited.xlsx"', sheet("Sheet1") firstrow clear
drop A
drop zkm-Q
sort year id line
gen line_id = _n
save `"${PATH_OUT_DATA}/dn_costfrequencydata.dta"', replace
log close
