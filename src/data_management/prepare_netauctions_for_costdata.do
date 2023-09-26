/*
	Prepare net auction data to be filled up with track cost data.
	Moreover, split awardings into single lines.
*/
capture log close
clear
set more off
set mem 2g
version 13
// Header do-File with path definitions, those end up in global macros.
include project_paths
log using `"${PATH_OUT_DATA}/log/`1'.log"', replace

* Load main data set.
use `"${PATH_OUT_DATA}/net_auctions.dta"', clear
capture drop _merge

****************************************
* Split auctions into single lines.    *
* Focus only on competitive awardings. *
****************************************

replace netz = regexr(netz, "\((.)+\)", "")
replace netz = subinstr(netz,",",";",.)
replace netz = subinstr(netz,"und",";",.) if id==3
replace netz = subinstr(netz,"sowie die",";",.) if id==3
replace netz = subinstr(netz,"?","-",.)
replace netz = subinstr(netz,"/Meiningen",",-Meiningen",.)
split netz, p(";" )

gen dup_no = .
forvalues i = 2(1)7 {
    replace dup_no = `i' if netz`i'!=""
    }
forvalues i = 2(1)7 {
    expand `i' if dup_no==`i'
    }
gen line=""
forvalues i = 1(1)7 {
    bysort id: replace netz`i'="" if _n!=`i'
	bysort id: replace line=netz`i' if _n==`i'
    }
sort year id line
drop netz1-netz7
replace line = trim(line)

* Generate within ID.
bysort id: gen id_within = _n


* Delete information on NKM and regional factors.
replace nkm=.
replace IK_GRUND=.
replace IK_REGFAK=.
replace EURproNKM=.

order id id_within year netz line nkm IK_GRUND IK_REGFAK EURproNKM zkm
saveold `"${PATH_OUT_DATA}/na_with_costdata.dta"', replace
capture saveold `"${PATH_OUT_DATA}/na_with_costdata.dta"', v(12) replace

drop netz
drop laufzeit-dup_no
saveold `"${PATH_OUT_DATA}/na_fill_costdata.dta"', replace
capture saveold `"${PATH_OUT_DATA}/na_fill_costdata.dta"', v(12) replace
log close
