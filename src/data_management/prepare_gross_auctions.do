/*
	Prepare gross auction data to be matched with track cost and characteristics data
	and exported to csv-format for reading into MATLAB.
	
	IMPORTANT: MAKE SURE SCALING IS EXACTLY THE SAME AS FOR THE NET AUCTION SAMPLE.

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
use `"${PATH_OUT_DATA}/ga_with_costdata.dta"', clear
capture drop _merge
drop nkm-EURproNKM

* Merge cost and track length data.
merge 1:1 id id_within using `"${PATH_OUT_DATA}/ga_costfrequencydata"'
drop if _merge==1
drop _merge

* Save merged data.
saveold `"${PATH_OUT_DATA}/ga_with_costdata_merged.dta"', replace
capture saveold `"${PATH_OUT_DATA}/ga_with_costdata_merged.dta"', v(12) replace

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
* Frequency of service per day assuming zkms are distributed equaly across lines within a contract (probably this variable does not make a lot of sense)
gen frequency_uniform = zkm_line_uniform / nkm * (1000000/365.25)

* Keep important variables.
drop if n_bidders==1
drop if netto==1
drop if EURproNKM==.
drop if nkm==.

* Rescale important regressors for gross auctions.
* BE CAREFUL TO MAKE SAME CHANGES FOR PREPARATION OF NET AUCTIONS.
* Compute winning bid scaled to full contract volume for a given line.
replace bid_win = bid_win * zkm_line_prop * laufzeit / 10
* Compute track access charges for full contract volume for a given line.
replace EURproNKM = EURproNKM / 10
* Adjust zkm to be on levle of full contract volume for a given line.
replace zkm_line_prop = zkm_line_prop 
* Scale total zkm of whole contract.
replace zkm = zkm  
* Generate total track access charges to be paid for full line.
gen AC_total = EURproNKM * zkm_line_prop * laufzeit / 10
* Compute margin, i.e. part of bid that is not needed to pay track access charges.
gen margin = bid_win - EURproNKM
sum margin, detail

* Check correlation pattern across different regressors.
corr(bid_win db_win n_bidders laufzeit zkm_line_prop EURproNKM untersttzungfahrzeugfinanzierung used_vehicles ///
 p_mode contract_mode elektro )
* Look at proposed set of cost regressors.
corr(laufzeit zkm_line_prop EURproNKM used_vehicles elektro )
sort id id_within zkm_line_prop
keep year id id_within nkm zkm zkm_line_prop laufzeit bid_win n_bidders db_win diesel EURproNKM margin used_vehicles frequency frequency_uniform db_not_old AC_total
order id id_within bid_win n_bidders db_win nkm EURproNKM zkm_line_prop frequency laufzeit zkm diesel 
order margin, last
order db_not_old, last
* Sort year to end of data set in order not to have to adjust MATLAB code.
order year, last
order AC_total, last
* Rescale regressors to have similar magnitude.
replace zkm = zkm / 10
replace laufzeit = laufzeit / 10
* Drop "Rasender Roland", because it is a very peculiar line that is not representative of our industry/model.
drop if id==17 | id==27
* Finally, add measure of number of potential bidders.
merge m:1 id using `"${PATH_OUT_DATA}/n_pot_ga"'
* Make sure lines we do not use anymore, but for which there is n_pot data, are dropped again.
drop if _merge==2
drop _merge
* Save file.
save `"${PATH_OUT_DATA}/ga_export.dta"', replace
* Export to CSV-file for MATLAB.
export delimited using `"${PATH_OUT_DATA}/ga_matlab.csv"', replace
log close
