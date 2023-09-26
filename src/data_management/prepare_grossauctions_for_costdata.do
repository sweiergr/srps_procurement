/*
	Prepare gross auction data to be filled up with track cost data.
	Moreover, split awarindgs into single lines.

*/

clear
set more off
set mem 2g
version 13
capture log close
// Header do-File with path definitions, those end up in global macros.
include project_paths
log using `"${PATH_OUT_DATA}/log/`1'.log"', replace

* Load main data set.
* Where does gross_auctions with nkmcost.data come from?
use `"${PATH_OUT_DATA}/gross_auctions.dta"', clear
capture drop _merge
****************************************
* Split auctions into single lines.    *
* Focus only on competitive awardings. *
****************************************

* Make sure that ), is split correctly to ,
* Currently this repetition of the command seems necessary, at some point automate this.
replace netz = regexr(netz, "\)\,", ",")
replace netz = regexr(netz, "\)\,", ",")
replace netz = regexr(netz, "\)\,", ",")
replace netz = regexr(netz, "\)\,", ",")
replace netz = regexr(netz, "\)\,", ",")
* Make sure that ); is split correctly to ;
* Currently this repetition of the command seems necessary, fix this at some point
replace netz = regexr(netz, "(.)\);(.)", ";")
replace netz = regexr(netz, "(.)\);(.)", ";")
replace netz = regexr(netz, "(.)\);(.)", ";")
replace netz = regexr(netz, "(.)\);(.)", ";")
replace netz = regexr(netz, "(.)\);(.)", ";")

* Remove comma inside parentheses for ID=8.
replace netz = subinstr(netz,", nur einzelne"," nur einzelne",.)
* Replace each comma with semicolon as split indicator.
replace netz = subinstr(netz,",",";",.)
* In 2003, for some lines / was used as split indicator.
replace netz = subinstr(netz,"/",";",.) if year==2003
split netz, p(";" )

gen dup_no = .
* Careful: these loop indices need adjustment depending on the maximum number of lines splitted.
forvalues i = 2(1)8 {
    replace dup_no = `i' if netz`i'!=""
    }
forvalues i = 2(1)8 {
    expand `i' if dup_no==`i'
    }
gen line=""
forvalues i = 1(1)8 {
    bysort id: replace netz`i'="" if _n!=`i'
	bysort id: replace line=netz`i' if _n==`i'
    }

sort  id line year

* Generate within ID.
by id: gen id_within = _n

drop netz1-netz8
replace line = trim(line)

* Delete information on NKM and regional factors.
replace nkm=.
replace IK_GRUND=.
replace IK_REGFAK=.
replace EURproNKM=.

order id id_within year netz line nkm IK_GRUND IK_REGFAK EURproNKM zkm
saveold `"${PATH_OUT_DATA}/ga_with_costdata.dta"', replace
capture saveold `"${PATH_OUT_DATA}/ga_with_costdata.dta"', v(12) replace

drop netz
drop laufzeit-dup_no

saveold `"${PATH_OUT_DATA}/ga_fill_costdata.dta"', replace
capture saveold `"${PATH_OUT_DATA}/ga_fill_costdata.dta"', v(12) replace
log close
