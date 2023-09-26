/*
	Import list of KKZ for each awarding.

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
import excel `"${PATH_IN_DATA}/vergabe_kkz.xlsx"', sheet("Sheet1") firstrow clear
destring kkz, replace
* Save file.
save `"${PATH_OUT_DATA}/vergabe_kkz_list.dta"', replace
log close
