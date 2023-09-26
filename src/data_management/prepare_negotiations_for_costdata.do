/*
	Prepare direct negotiations data to be filled up with track cost data.
	Moreover, split lines. 
    Note that this part of the sample is not used in the IJIO paper.

*/
clear
set more off
set mem 2g
capture log close
version 13
// Header do-File with path definitions, those end up in global macros.
include project_paths
log using `"${PATH_OUT_DATA}/log/`1'.log"', replace



* Load main data set.
use `"${PATH_OUT_DATA}/direct_negotiations.dta"', clear
*drop id
capture drop _merge

****************************************
* Split auctions into single lines.    *
* Focus only on competitive awardings. *
****************************************
replace netz = regexr(netz, "\((.)+\)", "")
replace netz = subinstr(netz,",",";",.)
replace netz = subinstr(netz,"?","-",.)
replace netz = subinstr(netz,"/",";",.) if year==2003
split netz, p(";" )

gen dup_no = .
forvalues i = 2(1)12 {
    replace dup_no = `i' if netz`i'!=""
    }
forvalues i = 2(1)12 {
    expand `i' if dup_no==`i'
    }
gen line=""
forvalues i = 1(1)12 {
    bysort id: replace netz`i'="" if _n!=`i'
	bysort id: replace line=netz`i' if _n==`i'
    }
sort year id line
drop netz1-netz12

* Generate within ID.
bysort id: gen id_within = _n
* Delete information on NKM and regional factors.
replace nkm=.
replace IK_GRUND=.
replace IK_REGFAK=.
replace EURproNKM=.
* Recast netz variable so that older Stata versions and R can read it.
recast str244 netz, force
order id id_within year netz line nkm IK_GRUND IK_REGFAK EURproNKM zkm
saveold `"${PATH_OUT_DATA}/negotiations_with_costdata.dta"', replace
capture saveold `"${PATH_OUT_DATA}/negotiations_with_costdata.dta"', v(12) replace
drop netz
drop laufzeit-dup_no
saveold `"${PATH_OUT_DATA}/negotiations_fill_costdata.dta"', replace
capture saveold `"${PATH_OUT_DATA}/negotiations_fill_costdata.dta"', v(12) replace
log close
