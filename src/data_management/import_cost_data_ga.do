/*
	For sample of gross auctions, 
	import track cost data from Excel file to Stata-file.

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
import excel `"${PATH_IN_DATA}/costdata_ga.xlsx"', sheet("Sheet1") firstrow clear

* Define yearly deflation factor for TPP based on Marktreport SPNV.
* Currently based on TPP growth rate between 2012 and 2014.
sca define deflate_factor = 1.0341
* Compute how many years between TPP-softwar year and auction.
gen years_deflate = (2013-year) * (TPP_2017==0) + (2017-year) * (TPP_2017==1)
* Deflate access charges.
gen EURproNKM_deflate = EURproNKM / (deflate_factor^years_deflate)
* Use deflated access charges instead of nominal ones.
replace EURproNKM = EURproNKM_deflate
* Drop variables that are not needed.
drop zkm-frequency
drop TPP_2017
drop EURproNKM_deflate
scalar drop deflate_factor
drop years_deflate
sort year id id_within line
saveold `"${PATH_OUT_DATA}/ga_costfrequencydata.dta"', replace
capture saveold `"${PATH_OUT_DATA}/ga_costfrequencydata.dta"', v(12) replace
log close
