/*
	Clean main data set and split it into gross and net auctions (and negotiations).
	This version is based on the confidential spnv_hauptbank.dta file.
	Some updates to the original data have been made after double-checking the original procurement documents:
	Subsidy is 5.72 EUR instead of 0.32 for net-ID 34.
	Subsidy is 4.34 EUR instead of 0.62 for net-ID 12.
*/
clear
capture log close
version 13
// Header do-File with path definitions, those end up in global macros.
include project_paths
log using `"${PATH_OUT_DATA}/log/`1'.log"', replace
set more off
set mem 2g


* Load main data set.
use `"${PATH_IN_DATA}/spnv_hauptbank.dta"', clear
capture drop _merge

* Incorporate vehicle subsidies into subsidy per zkm.
replace vehicle_subsidy = 0 if vehicle_subsidy==.
gen vs_prop = vehicle_subsidy / zkm / laufzeit
gen bestellerentgeltzkm_vs = bestellerentgeltzkm + vs_prop
drop vs_prop vehicle_subsidy

* Drop all variables that are not needed. 
drop heidelberg laufzeitbemerkung zkmalt politpreisbewertung preisbehandelt ///
	verfahrenbemerkung zugart traktionsart fahrzeug sonstigeszufahrzeugfin ///
	miopkmjahr sonstiges kommentare bemerkungallgemein bearbeitungsstand ///
	_ausschrbemerkung glaubwrdigkeit glaubwkommentar interessentenanzahl ///
	beiter* eigner_alt von bis eigner2 anteil2 eigner3 anteil3 ///
	eigner4 anteil4 eigner5 anteil5 insolvent eigner_neu datum* vergabejahr ///
	typ_neu datum* laufzeittage betende glz* merge* betreiber ///
	traktion* sbahn wettbewerb erfahrung_wbw_abs erfahrung_wbw_rel ///
	wbwantbl lerf lglz* lnkm lbieter llaufzeit frequenz altdb frq 
	
* Variables that can potentially be relevant for follow-up.
drop rcklaufbearbeitet bindefrist schlussterminfreinganggeboteteil vertragstypbemerkung ///
	 kostenunterlagen gesamtwertnetto anteilnettoin betrende standort zus* ///
	 gewstandort _bieter* anreizjn durchschnvorlauf vlauf dvl zkmbl pkmbl ///
	 wbw_anteil nebahnen ne lne mwjahr
drop durchschnittlichebefrderungskapa 

* Trash variables in original file.
drop v28 v45 v9 v10 v13 v14 v17 v18 v21 v22 v2 v1 j2 t t2

* Drop observations with important variables missing.
drop if bestellerentgeltzkm==.

* Recode variables.
encode anreiz, gen(incentives)
label define incentives_label 1 "Revenue incentives" 2 "Quality incentives" 3 "No information"
_strip_labels incentives
label values incentives incentives_label
replace incentives = 3 if incentives==.
drop anreiz
rename vertragsart contract_mode

* Rename variables.
rename bestellerentgeltzkm bid_win
rename bieteranzahl n_bidders
rename netzlnge track_length
rename gewinner winner
rename altanbieter incumbent
rename gebrauchtfahrzeuge used_vehicles
rename kostensteigerungenbernommen cost_increase
label var cost_increase "Cost increase absorption"
rename anteilkostensteigerungenbernomme cost_increase_share
label var cost_increase_share "Share cost increase absorption"
rename hauptland state_main
rename bundesland2 state_2
rename bundesland3 state_3

rename vergabeart p_mode
label var p_mode "Procurement mode"

rename db db_win
replace typ_alt = 0 if typ_alt==2
rename typ_alt db_incumbent
rename anteil1 share_inc

* Generate new variables.
* Here auction indicates any competitive awarding.
gen auction = 1 if p_mode < 7
replace auction = 0 if auction ==.

* Generate indicator for entrant being Altanbieter.
gen db_not_old = 0 if db_alt==1
* Fix some data coding errors based on values of other variables in the data set. 
replace db_not_old = 1 if db_not_old==.

* Look at statistics of winning bids.
bysort auction netto: sum bid_win
bysort p_mode netto db_win: sum bid_win

* Merge full list of IDs.
drop id
*merge m:1 jahr netz zkm using `"${PATH_OUT_DATA}/vergabe_id_list.dta"'
*drop _merge

rename jahr year
sort year netz laufzeit zkm
order year netz
drop if bid_win==.
save `"${PATH_OUT_DATA}/spnv_hauptbank_clean.dta"', replace

preserve
drop if n_bidders==. 
drop if p_mode==7
drop if netto!=0
gen id = _n
save `"${PATH_OUT_DATA}/gross_auctions.dta"', replace
restore

preserve
drop if n_bidders==. 
drop if p_mode==7
drop if netto!=1
* Merge ID variables to facilitate merging of reveneue track characteristics.
*merge 1:1 year netz using `"${PATH_IN_DATA}/na_ids.dta"'
capture drop _merge
drop if n_bidders==.
drop if bid_win==.
sort year netz laufzeit zkm
gen id = _n
save `"${PATH_OUT_DATA}/net_auctions.dta"', replace

restore
drop if p_mode!=7
gen id = _n
saveold `"${PATH_OUT_DATA}/direct_negotiations.dta"', replace
capture saveold `"${PATH_OUT_DATA}/direct_negotiations.dta"', v(12) replace
capture restore

log close
