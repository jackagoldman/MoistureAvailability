# Effects of Moisture Availability on Wildfire Burn Severity

This repository includes the data and R scripts to reproduce analyses and figures found in the article  _Moisture availability at short timescales influences wildfire burn severity in the boreal forest_ 

## Installation

The analyses were carried out with R version 4.1.3 and require the installation of a recent version of it.

## The following packages are required to run the scripts:

- tidyverse
- tidymodels
- caret
- ggplot2
- gbm
- gridExtra
- lubridate
- EnvStats
- ggpubr
- ggpattern
- viridis
- DALExtra

Below are instructions to load them all:

```
install.packages(c("tidyverse", "tidymodels", "caret", "ggplot", "gbm", "gridExtra", 
                 "lubridate", "EnvStats", "ggpubr", "ggpattern", "viridis", "DALExtra"))
```

## Details

To reproduce all the analyses run this code:

```
source("scripts/brt_analysis.R")
```

