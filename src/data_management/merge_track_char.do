/*
	Merge track characteristics that we use as regressors for revenue with main file.
	This file only merges track characteristics for net auctions!

*/
clear
capture log close
version 13
// Header do-File with path definitions, those end up in global macros.
include project_paths
log using `"${PATH_OUT_DATA}/log/`1'.log"', replace
set more off
set mem 2g
* Merge with net auctions sample.
use `"${PATH_OUT_DATA}/net_auctions.dta"', clear
capture drop _merge
merge 1:m id using `"${PATH_OUT_DATA}/vergabe_kkz_list.dta"'
* Destring KKZ variable.
capture destring kkz, replace
drop if _merge==2
drop _merge
sort id

merge m:m kkz using `"${PATH_OUT_DATA}/bevoelkerungkkz.dta"'
drop if _merge==2
drop _merge

sort id
destring y19*,force replace
destring y2*,force replace

gen AnzKreise=.
bysort id year laufzeit: replace AnzKreise=_N
bysort id year laufzeit: gen Einwohner1996=sum(y1996)
bysort id year laufzeit: gen Einwohner1998=sum(y1998)
bysort id year laufzeit: gen Einwohner1997=sum(y1997)
bysort id year laufzeit: gen Einwohner1999=sum(y1999)
bysort id year laufzeit: gen Einwohner2000=sum(y2000)
bysort id year laufzeit: gen Einwohner2001=sum(y2001)
bysort id year laufzeit: gen Einwohner2002=sum(y2002)
bysort id year laufzeit: gen Einwohner2003=sum(y2003)
bysort id year laufzeit: gen Einwohner2004=sum(y2004)
bysort id year laufzeit: gen Einwohner2005=sum(y2005)
bysort id year laufzeit: gen Einwohner2006=sum(y2006)
bysort id year laufzeit: gen Einwohner2007=sum(y2007)
bysort id year laufzeit: gen Einwohner2008=sum(y2008)
bysort id year laufzeit: gen Einwohner2009=sum(y2009)
bysort id year laufzeit: gen Einwohner2010=sum(y2010)
bysort id year laufzeit: gen Einwohner2011=sum(y2011)
bysort id year laufzeit: gen Einwohner2012=sum(y2012)
drop y19* y2*

gen einwohner=.
replace einwohner=Einwohner1996 if year==1996
replace einwohner=Einwohner1997 if year==1997
replace einwohner=Einwohner1998 if year==1998
replace einwohner=Einwohner1999 if year==1999
replace einwohner=Einwohner2000 if year==2000
replace einwohner=Einwohner2001 if year==2001
replace einwohner=Einwohner2002 if year==2002
replace einwohner=Einwohner2003 if year==2003
replace einwohner=Einwohner2004 if year==2004
replace einwohner=Einwohner2005 if year==2005
replace einwohner=Einwohner2006 if year==2006
replace einwohner=Einwohner2007 if year==2007
replace einwohner=Einwohner2008 if year==2008
replace einwohner=Einwohner2009 if year==2009
replace einwohner=Einwohner2009 if year==2010
replace einwohner=Einwohner2009 if year==2011
replace einwohner=Einwohner2009 if year==2012
* Fill track characteristic variable with number from earliest year available.
foreach var of varlist Einwohner1996-Einwohner2012{
	replace einwohner = `var' if einwohner==0
}

drop Einwohner*
merge m:m kkz using `"${PATH_IN_DATA}/kfz.dta"'
drop if _merge==2
drop _merge

bysort id year laufzeit: gen KfZins1996=sum(insgesamt1996)
bysort id year laufzeit: gen KfZins1998=sum(insgesamt1998)
bysort id year laufzeit: gen KfZins1997=sum(insgesamt1997)
bysort id year laufzeit: gen KfZins1999=sum(insgesamt1999)
bysort id year laufzeit: gen KfZins2000=sum(insgesamt2000)
bysort id year laufzeit: gen KfZins2001=sum(insgesamt2001)
bysort id year laufzeit: gen KfZins2002=sum(insgesamt2002)
bysort id year laufzeit: gen KfZins2003=sum(insgesamt2003)
bysort id year laufzeit: gen KfZins2004=sum(insgesamt2004)
bysort id year laufzeit: gen KfZins2005=sum(insgesamt2005)
bysort id year laufzeit: gen KfZins2006=sum(insgesamt2006)
bysort id year laufzeit: gen KfZins2007=sum(insgesamt2007)
bysort id year laufzeit: gen KfZins2008=sum(insgesamt2008)
bysort id year laufzeit: gen KfZins2009=sum(insgesamt2009)
bysort id year laufzeit: gen KfZins2010=sum(insgesamt2010)

bysort id year laufzeit: gen PKW1996=sum(pkw1996)
bysort id year laufzeit: gen PKW1998=sum(pkw1998)
bysort id year laufzeit: gen PKW1997=sum(pkw1997)
bysort id year laufzeit: gen PKW1999=sum(pkw1999)
bysort id year laufzeit: gen PKW2000=sum(pkw2000)
bysort id year laufzeit: gen PKW2001=sum(pkw2001)
bysort id year laufzeit: gen PKW2002=sum(pkw2002)
bysort id year laufzeit: gen PKW2003=sum(pkw2003)
bysort id year laufzeit: gen PKW2004=sum(pkw2004)
bysort id year laufzeit: gen PKW2005=sum(pkw2005)
bysort id year laufzeit: gen PKW2006=sum(pkw2006)
bysort id year laufzeit: gen PKW2007=sum(pkw2007)
bysort id year laufzeit: gen PKW2008=sum(pkw2008)
bysort id year laufzeit: gen PKW2009=sum(pkw2009)
bysort id year laufzeit: gen PKW2010=sum(pkw2010)

gen kfz=.
replace kfz=KfZins1996 if year==1996
replace kfz=KfZins1997 if year==1997
replace kfz=KfZins1998 if year==1998
replace kfz=KfZins1999 if year==1999
replace kfz=KfZins2000 if year==2000
replace kfz=KfZins2001 if year==2001
replace kfz=KfZins2002 if year==2002
replace kfz=KfZins2003 if year==2003
replace kfz=KfZins2004 if year==2004
replace kfz=KfZins2005 if year==2005
replace kfz=KfZins2006 if year==2006
replace kfz=KfZins2007 if year==2007
replace kfz=KfZins2008 if year==2008
replace kfz=KfZins2009 if year==2009
replace kfz=KfZins2010 if year==2010|year==2011

* Fill track characteristic variable with number from earliest year available.
foreach var of varlist KfZins1996-KfZins2010{
	replace kfz = `var' if kfz==0
}

drop KfZins*

gen pkw=.
replace pkw=PKW1996 if year==1996
replace pkw=PKW1997 if year==1997
replace pkw=PKW1998 if year==1998
replace pkw=PKW1999 if year==1999
replace pkw=PKW2000 if year==2000
replace pkw=PKW2001 if year==2001
replace pkw=PKW2002 if year==2002
replace pkw=PKW2003 if year==2003
replace pkw=PKW2004 if year==2004
replace pkw=PKW2005 if year==2005
replace pkw=PKW2006 if year==2006
replace pkw=PKW2007 if year==2007
replace pkw=PKW2008 if year==2008
replace pkw=PKW2009 if year==2009
replace pkw=PKW2010 if year==2010|year==2011

* Fill track characteristic variable with number from earliest year available.
foreach var of varlist PKW1996-PKW2010{
	replace pkw = `var' if pkw==0
}


drop PKW* kraftr* insgesamt*

merge m:m kkz using `"${PATH_OUT_DATA}/flaeche.dta"'
drop if _merge==2
drop _merge

bysort id year laufzeit: gen area1996=sum(flaeche1996)
bysort id year laufzeit: gen area1997=sum(flaeche1997)
bysort id year laufzeit: gen area1998=sum(flaeche1998)
bysort id year laufzeit: gen area1999=sum(flaeche1999)
bysort id year laufzeit: gen area2000=sum(flaeche2000)
bysort id year laufzeit: gen area2001=sum(flaeche2001)
bysort id year laufzeit: gen area2002=sum(flaeche2002)
bysort id year laufzeit: gen area2003=sum(flaeche2003)
bysort id year laufzeit: gen area2004=sum(flaeche2004)
bysort id year laufzeit: gen area2005=sum(flaeche2005)
bysort id year laufzeit: gen area2006=sum(flaeche2006)
bysort id year laufzeit: gen area2007=sum(flaeche2007)
bysort id year laufzeit: gen area2008=sum(flaeche2008)
bysort id year laufzeit: gen area2009=sum(flaeche2009)
bysort id year laufzeit: gen area2010=sum(flaeche2010)
bysort id year laufzeit: gen area2011=sum(flaeche2011)
bysort id year laufzeit: gen area2012=sum(flaeche2012)


gen area=.
replace area=area1996 if year==1996
replace area=area1997 if year==1997
replace area=area1998 if year==1998
replace area=area1999 if year==1999
replace area=area2000 if year==2000
replace area=area2001 if year==2001
replace area=area2002 if year==2002
replace area=area2003 if year==2003
replace area=area2004 if year==2004
replace area=area2005 if year==2005
replace area=area2006 if year==2006
replace area=area2007 if year==2007
replace area=area2008 if year==2008
replace area=area2009 if year==2009
replace area=area2010 if year==2010
replace area=area2011 if year==2011
replace area=area2012 if year==2012

* Fill track characteristic variable with number from earliest year available.
foreach var of varlist area1996-area2012{
	replace area = `var' if area==0
}


* For more recent data.
merge m:m kkz using `"${PATH_OUT_DATA}/einkommen.dta"'
drop if _merge==2
drop _merge
* For earlier income data.
merge m:m kkz using `"${PATH_IN_DATA}/einkommen_early.dta"'
drop if _merge==2
drop _merge

bysort id year laufzeit: gen income1996=sum(einkommen1996)
bysort id year laufzeit: gen income1997=sum(einkommen1997)
bysort id year laufzeit: gen income1998=sum(einkommen1998)
bysort id year laufzeit: gen income1999=sum(einkommen1999)
bysort id year laufzeit: gen income2000=sum(einkommen2000)
bysort id year laufzeit: gen income2001=sum(einkommen2001)
bysort id year laufzeit: gen income2002=sum(einkommen2002)
bysort id year laufzeit: gen income2003=sum(einkommen2003)
bysort id year laufzeit: gen income2004=sum(einkommen2004)
bysort id year laufzeit: gen income2005=sum(einkommen2005)
bysort id year laufzeit: gen income2006=sum(einkommen2006)
bysort id year laufzeit: gen income2007=sum(einkommen2007)
bysort id year laufzeit: gen income2008=sum(einkommen2008)
bysort id year laufzeit: gen income2009=sum(einkommen2009)
bysort id year laufzeit: gen income2010=sum(einkommen2010)
bysort id year laufzeit: gen income2011=sum(einkommen2011)
bysort id year laufzeit: gen income2012=sum(einkommen2012)
bysort id year laufzeit: gen income2013=sum(einkommen2013)


gen income=.
replace income=income1996 if year==1996
replace income=income1997 if year==1997
replace income=income1998 if year==1998
replace income=income1999 if year==1999
replace income=income2000 if year==2000
replace income=income2001 if year==2001
replace income=income2002 if year==2002
replace income=income2003 if year==2003
replace income=income2004 if year==2004
replace income=income2005 if year==2005
replace income=income2006 if year==2006
replace income=income2007 if year==2007
replace income=income2008 if year==2008
replace income=income2009 if year==2009
replace income=income2010 if year==2010
replace income=income2011 if year==2011
replace income=income2012 if year==2012
replace income=income2013 if year==2013


* Fill track characteristic variable with number from earliest year available.
foreach var of varlist income1996-income2013{
	replace income = `var' if income==0
}
replace income=. if income==0
replace kfz=. if kfz==0
replace einwohner=. if einwohner==0
rename einwohner population
replace area=. if area==0
replace pkw=. if pkw==0

drop pkw1* pkw2* flaeche1* flaeche2* area1* area2* einkommen1* einkommen2*
//v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 y1* y2* kreis

bysort id year laufzeit: gen anz=_N
bysort id year laufzeit: gen nr=_n
bysort id year laufzeit: drop if _n<anz

drop  anz nr

saveold `"${PATH_OUT_DATA}/netauctions_trackchar.dta"', replace
capture saveold `"${PATH_OUT_DATA}/netauctions_trackchar.dta"', v(12) replace
log close

