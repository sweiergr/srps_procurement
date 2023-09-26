/*
	Prepare net auction sample to be matched with track cost and characteristics data
	and export to csv-format for reading into MATLAB.
    
    Important:
    ENSURE THAT SCALING IS EXACTLY THE SAME AS FOR THE GROSS AUCTION SAMPLE!
*/

clear
capture log close
set more off
set mem 2g
version 13
* Header do-File with path definitions, those end up in global macros.
include project_paths
log using `"${PATH_OUT_DATA}/log/`1'.log"', replace
* Load main data set.
use `"${PATH_OUT_DATA}/netauctions_trackchar.dta"', clear
capture drop _merge
drop nkm-EURproNKM
order id
****************************************
* Split auctions into single lines.    *
* Focus only on competitive awardings. *
****************************************
* Manual fix for specific lines.
replace netz = subinstr(netz,", nur einzelne"," nur einzelne",.)
replace netz = subinstr(netz,"KBS 551, 585","KBS 551 585",.)
replace netz = subinstr(netz,"und",";",.) if id==3
replace netz = subinstr(netz,"sowie die",";",.) if id==3
replace netz = subinstr(netz,"?","-",.)
replace netz = subinstr(netz,"Stendal - WOB - MD","Stendal - WOB; WOB - MD",.) if id==29
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
* Replace each comma with semicolon as split indicator.
replace netz = subinstr(netz,",",";",.)
* Split netz variable at ;
split netz, p(";" )
gen dup_no = .
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
drop netz1-netz8
replace line = trim(line)
sort year id line

* Create line/within ID
bysort id: gen id_within = _n
order year id id_within line
drop netz

* Merge cost and track length data.
* na_costfrequencydata.dta is read in from manually collected cost data contained in an Excel sheet.
merge 1:m id id_within using `"${PATH_OUT_DATA}/na_costfrequencydata"'
drop if _merge==1
drop _merge
* Drop if nkm or EURproNKM data is missing.
drop if nkm==. | EURproNKM==.

* Save merged data.
saveold `"${PATH_OUT_DATA}/na_with_costdata_merged.dta"', replace
capture saveold `"${PATH_OUT_DATA}/na_with_costdata_merged.dta"', v(12) replace

* Compute frequency of service based on proportional distribution of zkm.
* Compute share of track of a line within a contract.
bysort id: egen total_nkm = sum(nkm)
gen nkm_share = nkm / total_nkm
* Compute line-specific zkm per year when distributing proportional to track length.
gen zkm_line_prop = zkm * nkm_share
* Compute line-specific zkm per year when distributing uniformly across lines within a contract.
bysort id: gen n_lines = _N
gen zkm_line_uniform = zkm / n_lines
* Frequency of service per day assuming frequency is identical across lines within a contract.
gen frequency = zkm_line_prop / nkm * (1000000/365.25)
* Frequency of service per day assuming zkms are distributed equaly across lines within a contract.
* (probably this variable does not make a lot of sense)
gen frequency_uniform = zkm_line_uniform / nkm * (1000000/365.25)

* Keep important variables.
drop if n_bidders==1
drop if netto==0

bysort id: egen avg_track_costs = mean(EURproNKM)

* Rescale important regressors for gross auctions.
* BE CAREFUL TO MAKE SAME CHANGES FOR PREPARATION OF GROSS AUCTIONS.
* Compute winning bid scaled to full contract volume for a given line.
replace bid_win = bid_win * zkm_line_prop * laufzeit / 10
* Compute track access charges for full contract volume for a given line.
replace EURproNKM = EURproNKM / 10
* Adjust zkm to be on level of full contract volume for a given line.
replace zkm_line_prop = zkm_line_prop 
* Scale total zkm of whole contract.
replace zkm = zkm 
* Generate total track access charges to be paid for full line.
gen AC_total = EURproNKM * zkm_line_prop * laufzeit / 10
* Compute margin, i.e. part of bid that is not needed to pay track access charges.
gen margin = bid_win - EURproNKM
sum margin, detail

* Compute some raw correlations and reduced form regressions.
* Check correlation between various regressors.
capture corr(bid_win zkm n_bidders EURproNKM population-log_inc_per_nkm)
* Check correlation pattern across different regressors.
corr(bid_win db_win n_bidders laufzeit zkm_line_prop EURproNKM untersttzungfahrzeugfinanzierung used_vehicles ///
 p_mode elektro )

* Look at proposed set of cost regressors.
corr(laufzeit zkm_line_prop EURproNKM used_vehicles elektro )

keep year id id_within zkm zkm_line_prop nkm laufzeit bid_win n_bidders db_win diesel EURproNKM margin used_vehicles ///
		population area income pkw frequency frequency_uniform db_not_old AC_total

* Rescale regressors to have similar magnitude.
replace zkm = zkm / 10
replace laufzeit = laufzeit / 10

* Fix missing values for used-vehicles variable.
* Naive approach: assume used vehicles permitted where not expicitly prohibited.
replace used_vehicles=1 if used_vehicles==.

* Drop Rasender Roland.
drop if id==20 | id==26
* Small line on non-DB netz.
drop if id==15
drop if id==9
* Drop net auction 44, because we only have partial data, and cannot recover much from original documents.
drop if id==44
* Same for line 43.
drop if id==43

order id id_within bid_win n_bidders db_win nkm EURproNKM zkm_line_prop frequency laufzeit zkm diesel used_vehicles
order year, last
order db_not_old, last
order AC_total, last
gen markup = bid_win / AC_total
sum markup, detail

* Add measure of number of potential bidders.
merge m:1 id using `"${PATH_OUT_DATA}/n_pot_na"'
* Make sure lines we do not use anymore, but for which there is n_pot data, are dropped again.
drop if _merge==2
drop _merge
* Save file.
save `"${PATH_OUT_DATA}/na_export.dta"', replace
* Export to CSV-MATLAB etc.
export delimited using `"${PATH_OUT_DATA}/na_matlab.csv"', replace
log close
