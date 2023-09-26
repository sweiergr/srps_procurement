/*
	Import track characteristics from txt-files, clean and save in Stata format.

*/

clear
capture log close
version 13
// Header do-File with path definitions, those end up in global macros.
include project_paths
log using `"${PATH_OUT_DATA}/log/`1'.log"', replace
set more off
set mem 2g

insheet using `"${PATH_IN_DATA}/Bevölkerung&Altersgruppen_Kreise.txt"'
drop if v48<=9 | v48>=528
destring v3, replace
destring v6, replace
destring v9, replace
destring v12, replace
destring v15, replace
destring v18, replace
destring v21, replace
destring v24, replace
destring v27, replace
destring v30, replace
destring v33, replace
destring v36, replace
destring v39, replace
destring v42, replace
destring v45, replace
drop v4 v5 v7 v8 v10 v11 v13 v14 v16 v17 v19 v20 v22 v23 v25 v26 v28 v29 v31 v32 v34 v35 v37 v38 v40 v41 v43 v44 v46 v47
rename v3 y2009
rename v6 y2008
rename v9 y2007
rename v12 y2006
rename v15 y2005
rename v18 y2004
rename v21 y2003
rename v24 y2002
rename v27 y2001
rename v30 y2000
rename v33 y1999
rename v36 y1998
rename v39 y1997
rename v42 y1996
rename v45 y1995
rename v1 kkz
destring kkz, replace
drop v2 v48

save `"${PATH_OUT_DATA}/Bevoelkerung.dta"', replace
clear

insheet using "kreise_vergabe.txt"
save "kreise_vergabe.dta", replace
clear

insheet using "Gebietsfläche_Kreise.txt"
rename v1 kkz
rename v2 kreis
rename v3 y2009
rename v4 y2008
rename v5 y2007
rename v6 y2006
rename v7 y2005
rename v8 y2004
rename v9 y2003
rename v10 y2002
rename v11 y2001
rename v12 y2000
rename v13 y1999
rename v14 y1998
rename v15 y1997
rename v16 y1996
rename v17 y1995

save "kreise_flaeche.dta", replace
clear

insheet using "kfz.txt"
drop v2
save "kfz.dta", replace
clear

insheet using "Einkommen.txt"
save "Einkommen_Kreise.dta", replace
clear

saveold `"${PATH_OUT_DATA}/spnv_hauptbank_w_track_char.dta"', replace
capture saveold `"${PATH_OUT_DATA}/spnv_hauptbank_w_track_char.dta"', v(12) replace
log close