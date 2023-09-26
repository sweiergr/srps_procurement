/*
	Analyze differences in winning bids across db_win and entrants for different 
	procurement formats.
	This file is only used for EDA, it does not generate any results reported in the IJIO paper.

*/
clear
capture log close
*log using db_win_entrant_bidspread.log, replace
set more off
set mem 2g
// Header do-File with path definitions, those end up in global macros.
include project_paths
log using `"${PATH_OUT_ANALYSIS}/log/`1'.log"', replace
* Load main data set.
use `"${PATH_IN_DATA}/spnv_hauptbank"', clear
capture drop _merge

* Rename some variables.
rename vergabeart p_mode
rename bieteranzahl n_bidders
rename bestellerentgeltzkm bid_win
gen auction = 1 if p_mode < 7
replace auction = 0 if auction ==.
rename db db_win

* Destring variables.
capture replace nkm=subinstr(nkm,",",".",.)
capture replace ik_grund=subinstr(ik_grund,",",".",.)
capture replace ik_regfak=subinstr(ik_regfak,",",".",.)
capture replace eurpronkm=subinstr(eurpronkm,",",".",.)
capture replace eurpronkm=subinstr(eurpronkm,"#DIV/0!","",.)
capture replace zkm=subinstr(zkm,",",".",.)
capture replace bid_win=subinstr(bid_win,",",".",.)
capture replace untersttzungfahrzeugfinanzierung=subinstr(untersttzungfahrzeugfinanzierung,",",".",.)
capture replace netzlnge=subinstr(netzlnge,",",".",.)
capture destring nkm, replace
capture destring ik_grund, replace
capture destring ik_regfak, replace
capture destring eurpronkm, replace
capture destring zkm, replace
capture destring bestellerentgeltzkm, replace
capture destring untersttzungfahrzeugfinanzierung, replace
capture destring netzlnge, replace

* Strip down data set to essential variables.
drop if p_mode==.
drop if n_bidders==. & p_mode<7

* Rename variables.
bysort auction netto: sum bid_win
bysort p_mode netto db_win: sum bid_win

* Compare winning bids for different auction formats:
* Plot Kernel Density of winning bid.
twoway (kdensity bid_win if auction==1 & netto==0 & db_win==0) (kdensity bid_win if auction==1 & netto==0 & db_win==1) (kdensity bid_win if auction==1 & netto==1 & db_win==0) (kdensity bid_win if auction==1 & netto==1 & db_win==1), title(Winning Bids (contracts)) legend(order(1 "Gross & Entrant Wins" 2 "Gross & db_win Wins" 3 "Net & Entrant Wins"  4 "Net & db_win wins"))
graph export `"${PATH_OUT_FIGURES}/winningbidcontracts.pdf"', replace
bysort netto: sum db_win if auction==1

* Before splitting contracts in lines.
* Do a formal t-test of whether winning bids are different across db_win and entrants.
* Pooling all competitive awardings.
ttest bid_win if auction==1, by(db_win) une
* By auction, competitive negotiation and non-competitive.
ttest bid_win if auction==1 & netto==0, by(db_win) une
ttest bid_win if auction==1 & netto==1, by(db_win) une
ttest bid_win if p_mode==1, by(db_win) une
ttest bid_win if p_mode==1 & netto==0, by(db_win) une
ttest bid_win if p_mode==1 & netto==1, by(db_win) une
ttest bid_win if p_mode==2, by(db_win) une
ttest bid_win if p_mode==2 & netto==0, by(db_win) une
ttest bid_win if p_mode==2 & netto==1, by(db_win) une
ttest bid_win if p_mode==7 & netto==1, by(db_win) une

****************************************
* Split auctions into single lines.    *
* Focus only on competitive awardings. *
****************************************

capture gen id= _n
replace netz = regexr(netz, "\((.)+\)", "")
replace netz = subinstr(netz,",",";",.)
split netz, p(";")
gen dup_no = .
forvalues i = 2(1)10 {
    replace dup_no = `i' if netz`i'!=""
    }
forvalues i = 2(1)10 {
    expand `i' if dup_no==`i'
    }
gen line=""
forvalues i = 1(1)10 {
    bysort id: replace netz`i'="" if _n!=`i'
	bysort id: replace line=netz`i' if _n==`i'
    }
sort netz
drop netz*
* Split train kilometers naively from contract level to line level.
replace dup_no=1 if dup_no==.
gen zkm_line = zkm / dup_no
bysort netto: tab db_win
* Look at available observations.
bysort p_mode netto db_win: sum bid_win




*Compare winning bids for different auction formats:
* Plot Kernel Density of winning bid.
twoway (kdensity bid_win if auction==1 & netto==0 & db_win==0) (kdensity bid_win if auction==1 & netto==0 & db_win==1) (kdensity bid_win if auction==1 & netto==1 & db_win==0) (kdensity bid_win if auction==1 & netto==1 & db_win==1), title(Kernel density of winning bids) xscale(r(0 10)) xtitle("Subsidy per train-km (in EUR)") ytitle("") legend(order(1 "Gross & entrant wins" 2 "Gross & incumbent wins" 3 "Net & entrant wins"  4 "Net & incumbent wins"))
graph export `"${PATH_OUT_FIGURES}/winningbidlines.pdf"', replace
bysort netto: sum db_win if auction==1

***********************************
* NOW FOR AWARDINGS SPLIT BY LINES
***********************************
* Do a formal t-test of whether winning bids are different across db_win and entrants.
* Pooling all competitive awardings.
ttest bid_win if auction==1, by(db_win) une
* By auction, competitive negotiation and non-competitive.
ttest bid_win if auction==1 & netto==0, by(db_win) une
ttest bid_win if auction==1 & netto==1, by(db_win) une
ttest bid_win if p_mode==1, by(db_win) une
ttest bid_win if p_mode==1 & netto==0, by(db_win) une
ttest bid_win if p_mode==1 & netto==1, by(db_win) une
ttest bid_win if p_mode==2, by(db_win) une
ttest bid_win if p_mode==2 & netto==0, by(db_win) une
ttest bid_win if p_mode==2 & netto==1, by(db_win) une
ttest bid_win if p_mode==7 & netto==1, by(db_win) une

* Check whether gross and net contracts are similar in contract characteristics.
local track_char "laufzeit zkm miopkmjahr auction untersttzungfahrzeugfinanzierung anreiz traktionsart zugart gebrauchtfahrzeuge glaubwrdigkeit jahr IK_GRUND IK_REGFAK diesel EURproNKM frq elektro"
foreach var of varlist zkm gebrauchtfahrzeuge IK_GRUND IK_REGFAK diesel{
	disp("Track characteristic to be tested on equality:  `var'")
	ttest `var' if p_mode==1,by(netto)
}

bysort db_win netto: sum bid_win if p_mode==7
ttest bid_win if p_mode==7 & netto==1, by(db_win) une

tab n_bidders if p_mode<7 & netto==0 & n_bidders>1 & bid_win!=.

* Check how many observations we have after splitting up lines.
* Summarize auctiosn and competitive negotiations.
bysort netto db_win: sum bid_win if p_mode<7
* Check number of observations for direct negotiations.
bysort netto db_win: sum bid_win if p_mode==7

* Plot empirical CDFs of winning bids.
* Winning bids by db_win.
cdfplot(bid_win) if p_mode<7 & bid_win!=. & n_bidders!=. & netto==0 & db_win==1
graph export `"${PATH_OUT_FIGURES}/cdf_bid_win_db.pdf"', replace

* Winning bids by entrant.
cdfplot(bid_win) if p_mode<7 & bid_win!=. & n_bidders!=. & netto==0 & db_win==0
graph export `"${PATH_OUT_FIGURES}/cdf_bid_win_entrant.pdf"', replace

* There seems to be a lot of heterogeneity in contracts awarded in gross or net format.
* Check for observed auction heterogeneity in gross auctions.
ttest zkm if auction==1 & netto==0 & bid_win!=., by(db_win) une
ttest gebrauchtfahrzeuge if auction==1 & netto==0 & bid_win!=., by(db_win) une
ttest elektro if auction==1 & netto==0 & bid_win!=., by(db_win) une
ttest diesel if auction==1 & netto==0 & bid_win!=., by(db_win) une



drop if bid_win==. | n_bidders==.
sort id dup_no
drop if p_mode==7
drop if n_bidders<2
drop if netto==1
order id dup_no db_win bid_win n_bidders zkm_line zkm
export delimited using `"${PATH_OUT_DATA}/grossauctions_2.csv"', replace

log close
