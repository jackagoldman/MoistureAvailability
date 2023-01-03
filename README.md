# MoistureAvailability
Analysis for: Moisture availability at short timescales influences wildfire burn severity in the boreal  forest

This repository includes the data and R scripts to reproduce analyses and figures found in the article _ _Moisture availability at short timescales influences wildfire burn severity in the boreal forest_ _

## Installation

The analyses were carried out with R version 4.1.3 and require the installation of a recent version of it.

## The following packages are required to run the scripts:

-tidyverse
-tidymodels
-caret
-ggplot
-gbm

Below are instructions to load them all:

```
install.packages(c("tidyverse", "tidymodels", "caret", "ggplot", "gbm"))
```

## Details

To reproduce all the analyses run this code:

```
source("scripts/NW_analysis..")
```

