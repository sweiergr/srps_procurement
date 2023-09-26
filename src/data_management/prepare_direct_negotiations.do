/*
	Prepare direct negotiations data to have filled up with track cost data.
	Note that this part of the sample is not used in the IJIO paper.

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
use `"${PATH_OUT_DATA}/negotiations_with_costdata.dta"', clear
capture drop _merge
drop nkm-eurpronkm
sort year id line
gen line_id = _n
* Merge cost and track length data. 
* dn_costfrequencydata.dta is created manually by merging track data with cost data gathered manually.
merge 1:1 line_id using `"${PATH_OUT_DATA}/dn_costfrequencydata"'
saveold `"${PATH_OUT_DATA}/dn_with_costdata_merged.dta"', replace
capture saveold `"${PATH_OUT_DATA}/dn_with_costdata_merged.dta"', v(12) replace

* Compute frequency of service based on proportional distribution of zkm.
bysort id: egen total_nkm = sum(nkm)
gen nkm_share = nkm / total_nkm
gen zkm_line_prop = zkm * nkm_share
bysort id: gen n_lines = _N
gen zkm_line_uniform = zkm / n_lines
gen frequency = zkm_line_prop / nkm * (1000000/365.25)
* This variable probably does not make a lot of sense. Be careful when using it!
gen frequency_uniform = zkm_line_uniform / nkm * (1000000/365.25)

* Check correlation pattern across different regressors.
corr(bid_win db_win n_bidders laufzeit zkm_line_prop EURproNKM untersttzungfahrzeugfinanzierung used_vehicles ///
 p_mode contract_mode elektro )
* Look at proposed set of cost regressors.
corr(laufzeit zkm_line_prop EURproNKM used_vehicles elektro )
gen margin = bid_win - EURproNKM
sum margin

* Run logit of DB winning on regressors.
logit db_win zkm laufzeit EURproNKM used_vehicles
* Run reduced form regression of winning bid on track characteristics.
reg bid_win  n_bidders laufzeit zkm_line_prop EURproNKM used_vehicles diesel
* With total contract number of zkm.
reg bid_win  n_bidders laufzeit zkm used_vehicles diesel EURproNKM
* "Correct" negative margins. Double-check this in the data.
replace margin=0.5 if margin<0

* Run reduced form regression of winning bid on track characteristics.
reg margin n_bidders laufzeit zkm_line_prop used_vehicles diesel
* With total contract number of zkm.
reg margin  n_bidders laufzeit zkm used_vehicles diesel 

* Keep important variables.
drop if n_bidders==1
drop if netto==1
rename id contract_id
gen line_id = _n
keep contract_id line_id zkm zkm_line_prop laufzeit bid_win n_bidders db_win diesel EURproNKM margin used_vehicles
bysort contract_id: egen avg_track_costs = mean(EURproNKM)
order contract_id line_id bid_win margin n_bidders db_win zkm laufzeit diesel EURproNKM zkm_line_prop avg_track_costs

* Rescale regressors to have similar magnitude.
replace zkm = zkm / 10
replace laufzeit = laufzeit / 10
* Export to CSV-MATLAB etc.
export delimited using `"${PATH_OUT_DATA}/dn_matlab.csv", replace
log close
