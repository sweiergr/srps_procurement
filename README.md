# Bidder Asymmetries in Procurement Auctions: Efficiency versus Information
This document describes the replication package that accompanies *Bidder Asymmetries in Procurement Auctions: Efficiency versus Information - Evidence from Railway Passenger Services* by Christoph Carnehl and Stefan Weiergraeber (IJIO, 2023).

## Overview
The code in this replication package constructs all tables and figures in the main text and the online appendix from the original data sources using a combination of Python, R, Stata, and MATLAB.

The code is structured based on a template for reproducible research projects in Economics by Gaudecker (2019). The automation is pre-configured by  describing the research workflow as a dependent acyclic graph using Waf.

In order to replicate all figures and tables referenced in the paper you need to:
- Install all the required software and packages (see *Software Requirements* below). If not already the case, add all executables for Python, R, Stata, and MATLAB to your system path so that waf can access the software from the command line.
- Copy the confidential data files, which are not provided as part of this package, into ```src/original_data```.
- Navigate to the root folder of the code repository.
- Configure waf using the command ```python waf.py configure```.
- Let waf build the project using the command ```python waf.py build```. The file ```waf.py``` along with the ```wscript```-files in each subfolder and the code in folder ```.waf``` serve as a master file that runs all the programs for data management, analysis, and final formatting in the correct order. In addition, waf will create the corresponding folder structure within a ```bld```-folder in the root-directory of the replication package. Details about this setup are provided by Gaudecker (2019). Therefore, you should not run any of the programs manually before running the waf-program. Once the folder structure and the associated project path files in ```bld``` are created, you can interactively rerun any of the programs contained in this repository and inspect all of its output. When running single files in isolation please set the working directory of your Stata/MATLAB/Python/R executable to ```bld``` to ensure that the automatically generated absolute path names can be accessed by the programs.
    
Note that waf will run independent programs in parallel. On some machines this could lead to an inefficient overuse of hardware and in rare circumstances crash the system. In order to circumvent this, waf can be instructed to run all files serially by using the ```-j1```-flag: ```python waf.py build -j1```. 

You should expect the code to run for about 7 days on a modern desktop computer.

## Data Availability Statement
All data used in this paper is confidential and was obtained from *Nahverkehrsberatung Suedwest*.

## Computational Requirements
### Software Requirements
The following software  is necessary to build the project. The code does not install these packages. Please install each of the software and required packages manually before running the code.
- Stata (code was last run using version 17)
    - estout
- Python (code was last run using Anaconda Python 3.8.5, which should come with most of the required packages by default.) **IMPORTANT: The waf-template will not run under Python 3.11**
- Matlab (code was last run using Release 2020a)
- R (code was last run using version 4.2)
    - apsrtable
    - dummies
    - forcats
    - haven
    - hrbrthemes
    - plm 
    - qwraps2 
    - readxl 
    - sandwich
    - stargazer
    - tidyverse 

### Memory and Runtime Requirements
The code was last run on a 6-core Intel-based iMac with MacOS version 10.15.7. Computation took approximately 2 days.

## Description of Code
- All program files contain an introductory docstring that explains the purpose and content of the respective file.
- All original data source files should be stored in src/original_data. The data management code reads all data source files from this location.
- The programs in ```src/data_management``` clean, reshape, and combine the different data files for both the reduced form regressions and the structural estimation. ```src/data_management/wscript``` describes the detailed dependency structure and explains which output files are generated by which program. The cleaned data output files are stored in ```bld/out/data```.
- The programs in src/analysis run the main analysis consisting of reduced form regressions and the structural estimation. ```src/analysis/wscript``` describes the detailed dependency structure and explains which output files are generated by which program. Most of the estimation output is stored in unformatted form in ```bld/out/analysis```. Some reduced form regression tables are directly formatted in Stata and are saved in ```bld/out/tables```.
- The programs in ```src/final``` read the estimation output from ```bld/out/analysis``` and format the final LaTeX-tables for the structural estimation results and the counterfactual results. In addition, the scripts that generate the figures are stored in this folder. ```src/final/wscript``` describes the detailed dependency structure and explains which output files are generated by which program. The figures are stored in ```bld/out/figures```.
- The folder ```src/model_specs``` contains several configuration files for the estimation, such as optimizer options, and starting values for the parameters that were obtained using an exploratory broad search radius using ```src/analysis/ars.m```.
- Note that the file ```waf.py``` along with the ```wscript```-files in each subfolder and the code in folder ```.waf``` serves as a master file that runs all the programs for data management, analysis, and final formatting in the correct order. In order to build the project, please execute the following:
1. Navigate to the root folder of the code repository.
2. Configure waf using the command ```python waf.py configure```
3. Build the project using the command ```python waf.py build```. This
step will also create the corresponding folder structure within a ```bld```-folder in the root-directory of the replication package. Details about this setup are provided by Gaudecker (2019).
- Please note that the waf-template is currently incompatible with Python 3.11.
- You should not run any of the programs manually before running the waf-program as described above. Once the folder structure and the associated project path files in bld are created on your computer, you can interactively rerun any of the programs contained in this repository and inspect all of its output. When running single files in isolation please set the working directory of your Stata/MATLAB/Python/R executable to bld to ensure that the automatically generated absolute path names can be accessed by the programs.

## References
- Carnehl, Christoph and Weiergraeber, Stefan (2023). Bidder Asymmetries in Procurement Auctions: Evidence versus Information - Evidence from Railway Passenger Services. International Journal of Industrial Organization: Volume 87, March 2023, 102902.
- Gaudecker, Hans-Martin von (2019). Templates for Reproducible Research
Projects in Economics (v0.1). https://doi.org/10.5281/zenodo.2533241/.