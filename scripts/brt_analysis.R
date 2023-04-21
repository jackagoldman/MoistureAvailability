### Stochastic Gradient Boosted Regression Tree Analysis ###

# Author: Jack A. Goldman

# Date: 2023-01-04

#required packages
library(caret)
library(gbm)
library(dplyr)
library(tidyverse)
library(ggplot2)
library(tidymodels)
library(readr)


# Data preparation ---------------------

### DATA ###

#data_tlm <- read_csv2("data/TimeLagMoisture_data.csv")

#climate data
data_tlm <- read_csv2("data/temporal-lag-moisture-id.csv")

#covariates
covariates <- read_csv2("data/nwo-managedarea-covariates.csv")

#select covariates of interest
covariates <- select(covariates, c("raster_id", "per_fc", "yearf"))

#join covariates to climate data and remove id
data_tlm <- data_tlm %>% 
  left_join(covariates, by = "raster_id") %>% 
  select(-c(2))

# Remove first column
data_tlm_clean <- data_tlm %>%  
  select(-c(1))

#split data set into Extreme only and Median only

med_tlm <- data_tlm_clean %>% 
  select(-c(2))

ext_tlm <- data_tlm_clean %>% 
  select(-c(1))

### Split data into test vs. training ###

# Extreme
set.seed(429)#set seed for reproducibility


ext_split <- initial_split(ext_tlm)
ext_train <- training(ext_split)
ext_test  <- testing(ext_split)

# Median
set.seed(429)##set seed for reproducibility


med_split <- initial_split(med_tlm)
med_train <- training(med_split)
med_test  <- testing(med_split)


#First thing is to set up the defaults of the model. 
#These are the metrics that will be used for both trees 
#we use RMSE. 
#We also set up the control grid for tuning. 
#Additionally, we set up the cross-validation process for our training model 
#using 10-fold Cross-Validation

# Extreme wildfire burn severity -----------------------------------------
#set up defaults
mset <- metric_set(rmse) 
control <- control_grid(save_workflow = TRUE,
                        save_pred = TRUE,
                        extract = extract_model) # grid for tuning


set.seed(429) #set seed for reproducibility

ext_folds <- vfold_cv(ext_train, v = 10)


#Next step is to build up the recipe, or our model fit
ext_fit = recipe(RBR_quant ~., data = ext_train) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_corr(all_numeric_predictors(), threshold = .8)

#Now to specify the model
ext_tune_spec = 
  boost_tree(trees =  1000,
             tree_depth = tune(),
             learn_rate = tune(),
             min_n = tune(),
             mtry = tune()) %>% 
  set_mode("regression") %>% 
  set_engine("xgboost", counts = FALSE)

# set workflow

ext_wf <- workflow() %>%
  add_recipe(ext_fit) %>% 
  add_model(ext_tune_spec)


#Use 10-fold cross-validation to evaluate the model with different hyperparameters
set.seed(429) #set seed for reproducibility


ext_tune<-  ext_wf %>% 
  tune_grid(resample = ext_folds,
            metrics = mset,
            control = control,
            grid = crossing( 
              tree_depth = c(2,4,5,3),
              learn_rate = c(0.0001,0.005, 0.025),
              min_n = c(10, 15, 20),
              mtry = c(0.5)))

#Use best tuning specifications to fit model to training data.
set.seed(429) #set seed for reproducibility

ext_fit <- ext_wf %>% 
  finalize_workflow(select_best(ext_tune)) %>% 
  fit(ext_train)

#Check out accuracy on testing dataset to see if we overfitted.
set.seed(429)#set seed for reproducibility

ext_fit %>%
  augment(ext_test, type.predict = "response") %>% 
  rmse(RBR_quant, .pred) # 14 points off

# Set resample control
keep_pred <- control_resamples(save_pred = TRUE, save_workflow = TRUE)

set.seed(429)#set seed for reproducibility

# 10-fold cross validation on training to get RMSE and StDev
ext_res <- 
  ext_fit%>%  
  fit_resamples(resamples = ext_folds, control = keep_pred)

collect_metrics(ext_res)


#resample test vs. train to evaluate model performance
set.seed(429)
cvFolds_exttest <- ext_test %>% vfold_cv(10)
set.seed(429)
exttest_fit <- ext_wf %>% 
  finalize_workflow(select_best(ext_tune)) %>% 
  fit(ext_test)
set.seed(429)
exttest_res <-  exttest_fit %>% fit_resamples(resamples = cvFolds_exttest, control = keep_pred)
exttest_res %>%  collect_metrics() # testing
ext_res %>%  collect_metrics(summarize = TRUE) # training

#Extract and visualize variable importance
ext_imp <- xgboost::xgb.importance(model = extract_fit_engine(ext_fit))

ext_imp %>%
  mutate(Feature = fct_reorder(Feature, Gain)) %>% 
  dplyr::slice_max(Gain, n = 10) %>% 
  ggplot(aes(Gain, Feature, fill = Feature)) +
  geom_col() + scale_fill_viridis_d()+ theme_bw()  +
  theme(legend.position="none") + 
  labs(title = "Burn Severity Extremes", y = "Predictor") +
  theme(plot.title = element_text(hjust = 0.5))

# Median wildfire Burn Severity -----------------------------------------

#set up defaults
mset <- metric_set(rmse) # metric is accuracy
control <- control_grid(save_workflow = TRUE,
                        save_pred = TRUE,
                        extract = extract_model) # grid for tuning


#10 fold cross val
cvFolds_med <- med_train %>% vfold_cv(10)

# build recipe
med_rec = recipe(RBR_median ~., data = med_train) %>% 
  step_normalize(all_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_corr(all_numeric_predictors(), threshold = .8)


# specify BRT model
med_tune_spec = 
  boost_tree(trees =  1000,
             tree_depth = tune(),
             learn_rate = tune(),
             min_n = tune(),
             mtry = tune()) %>% 
  set_mode("regression") %>% 
  set_engine("xgboost", counts = FALSE)


# set workflow

med_wf <- workflow() %>%
  add_recipe(med_rec) %>% 
  add_model(med_tune_spec)


# use CV to evaluate the model with different hyperparameters
set.seed(429)
med_tune<-  med_wf %>% 
  tune_grid(resamples = cvFolds_med,
            metrics = mset,
            control = control,
            grid = crossing( 
              tree_depth = c(6, 10),
              learn_rate = c(0.001, 0.005, 0.025),
              min_n = c(10, 15, 20 ),
              mtry = c(0.5)))

#Use best tuning specifications to fit model to training data.
set.seed(429)
med_fit <- med_wf %>% 
  finalize_workflow(select_best(med_tune)) %>% 
  fit(med_train)

#Check out accuracy on testing dataset to see if we overfitted.
set.seed(429)
med_fit %>%
  augment(med_test, type.predict = "response") %>% 
  rmse(RBR_median, .pred)

# set the resample control
keep_pred <- control_resamples(save_pred = TRUE, save_workflow = TRUE)

# 10-fold cross validation on training to get RMSE and StDev
set.seed(429)
med_res <- 
  med_fit%>%  
  fit_resamples(resamples = cvFolds_med, control = keep_pred)


#resample test vs. train to evaluate model performance
set.seed(429)
cvFolds_medtest <- med_test %>% vfold_cv(10)
set.seed(429)
medtest_fit <- med_wf %>% 
  finalize_workflow(select_best(med_tune)) %>% 
  fit(med_test)
set.seed(429)
medtest_res <-  medtest_fit %>% fit_resamples(resamples = cvFolds_medtest, 
                                              control = keep_pred)

medtest_res %>%  collect_metrics() # testing
med_res %>%  collect_metrics(summarize = TRUE)# training

# Extract and plot variable importance

med_imp <- xgboost::xgb.importance(model = extract_fit_engine(med_fit))

med_imp %>%
  mutate(Feature = fct_reorder(Feature, Gain)) %>% 
  dplyr::slice_max(Gain, n = 10) %>% 
  ggplot(aes(Gain, Feature, fill = Feature)) +
  geom_col() + scale_fill_viridis_d()+ theme_bw()  +
  theme(legend.position="none") + 
  labs(title = "Median Burn Severity within AOU", y = "Predictor") +
  theme(plot.title = element_text(hjust = 0.5)) 



# Figures included are solely from results of the analysis

#required packages
library(gridExtra)
library(lubridate)
library(ggplot2)
library(EnvStats)
library(ggpubr)
library(ggpattern)
library(viridis)
library(DALEXtra)

# Figure 3 Relative Importance top 10 -------------
#extreme plot - bw
ex.im.plot.bw<-ext_imp %>%
  mutate(Feature = fct_reorder(Feature, Gain)) %>% 
  dplyr::slice_max(Gain, n = 10) %>% 
  mutate(Climate_Metrics = case_when(
    grepl("C", Feature) ~ "Climate Moisture Index",
    grepl("R", Feature) ~ "Relative Humidity",
    grepl("T", Feature) ~"Maximum Temperature"
  )) %>% 
  mutate(Temporal_Dynamics = case_when(
    grepl("m", Feature) ~ "Monthly",
    grepl("y", Feature) ~ "Yearly"
  )) %>% 
  ggplot(aes(Gain, Feature, fill = Climate_Metrics), color = "black") +
  geom_col_pattern(
    aes(pattern = Temporal_Dynamics),
    colour = "black",
    pattern_fill = "black",
    pattern_angle = 45,
    pattern_density = 0.05,
    pattern_spacing = 0.03,
    position = position_dodge2(preserve = 'single')) +
  scale_pattern_manual(values = c("none", "stripe"), "Temporal Dynamics",
                       guide = guide_legend(override.aes = list(fill = "white"))) +
  theme(text = element_text(size = 10)) +
  
  scale_fill_manual(values = c("#808080", "#C0C0C0", "#313335"), "Climate Metrics", 
                    guide = guide_legend(override.aes = list(pattern = "none")))+
  scale_x_continuous(labels = scales::percent) + 
  theme_bw()+
  labs(title = "Extreme Burn Severity",  x = "Relative Influence (%)", y = "Climate Predictor") +
  theme(plot.title = element_text(hjust = 0.5))+
  theme(panel.background = element_blank())


#medianplot - bw
md.im.plot.bw <- med_imp %>%
  mutate(Feature = fct_reorder(Feature, Gain)) %>% 
  dplyr::slice_max(Gain, n = 10) %>% 
  mutate(Climate_Metrics = case_when(
    grepl("C", Feature) ~ "Climate Moisture Index",
    grepl("R", Feature) ~ "Relative Humidity",
    grepl("T", Feature) ~"Maximum Temperature"
  )) %>% 
  mutate(Temporal_Dynamics = case_when(
    grepl("m", Feature) ~ "Monthly",
    grepl("y", Feature) ~ "Yearly"
  )) %>% 
  ggplot(aes(Gain, Feature, fill = Climate_Metrics), color = "black") +
  geom_col_pattern(
    aes(pattern = Temporal_Dynamics),
    colour = "black",
    pattern_fill = "black",
    pattern_angle = 45,
    pattern_density = 0.05,
    pattern_spacing = 0.03,
    position = position_dodge2(preserve = 'single')) +
  scale_pattern_manual(values = c("none", "stripe"), "Temporal Dynamics",
                       guide = guide_legend(override.aes = list(fill = "white"))) +
  theme(text = element_text(size = 10)) +
  
  scale_fill_manual(values = c("#808080", "#C0C0C0", "#313335"), "Climate Metrics", 
                    guide = guide_legend(override.aes = list(pattern = "none")))+
  scale_x_continuous(labels = scales::percent) + 
  theme_bw()+
  labs(title = "Median Burn Severity",  x = "Relative Influence (%)", y = "Climate Predictor") +
  theme(plot.title = element_text(hjust = 0.5))+
  theme(panel.background = element_blank())

#create multiplot
ggarrange(md.im.plot.bw, ex.im.plot.bw, 
          labels = c("A", "B"),
          ncol = 2, nrow = 1, widths = c(1,1),
          common.legend = TRUE, legend="bottom")

# Figure 4 Percent Importance Temporal-----
#median
rel.per.imten = ext_med %>%
  mutate(Feature = 
           fct_reorder(Feature, Gain)) %>% 
  dplyr::slice_max(Gain, n = 10) %>% 
  mutate(Climate_Metrics = case_when(
    grepl("C", Feature) ~ "Climate Moisture Index",
    grepl("R", Feature) ~ "Relative Humidity",
    grepl("T", Feature) ~"Maximum Temperature"
  )) %>% 
  mutate(Temporal_Dynamics = case_when(
    grepl("yCsum", Feature) ~ "Yearly",
    grepl("yRmean", Feature) ~ "Yearly",
    grepl("mCsum", Feature) ~ "Monthly",
    grepl("mRmean", Feature) ~ "Monthly",
    grepl('m', Feature) ~ "Monthly",
    grepl('y', Feature) ~ "Yearly"
  )) %>% 
  group_by(Temporal_Dynamics) %>% 
  summarise(total_inf = sum(Gain)) %>% 
  ggplot(aes(y= total_inf, x = Temporal_Dynamics)) +
  geom_bar(stat = "identity", fill = "darkgrey", colour = "black", alpha = 0.7,
           width = 0.6)+
  scale_y_continuous(labels = scales::percent) + 
  theme_bw() + 
  labs(title ="Median Burn Severity", 
       x ="Temporal Dynamics", 
       y ="Relative Influence (%)") +
  theme(plot.title = element_text(hjust = 0.5, size = 10))+ 
  scale_x_discrete(expand = expansion(add = c(0.5, 0.5)))+
  theme(plot.margin = margin(2,2,2,2, "mm"))

#Extreme
rel.per.iqten = ext_imp %>%
  mutate(Feature = 
           fct_reorder(Feature, Gain)) %>% 
  dplyr::slice_max(Gain, n = 10) %>% 
  mutate(Climate_Metrics = case_when(
    grepl("C", Feature) ~ "Climate Moisture Index",
    grepl("R", Feature) ~ "Relative Humidity",
    grepl("T", Feature) ~"Maximum Temperature"
  )) %>% 
  mutate(Temporal_Dynamics = case_when(
    grepl("yCsum", Feature) ~ "Yearly",
    grepl("yRmean", Feature) ~ "Yearly",
    grepl("mCsum", Feature) ~ "Monthly",
    grepl("mRmean", Feature) ~ "Monthly",
    grepl('m', Feature) ~ "Monthly",
    grepl('y', Feature) ~ "Yearly"
  )) %>% 
  group_by(Temporal_Dynamics) %>% 
  summarise(total_inf = sum(Gain)) %>% 
  ggplot(aes(y= total_inf, x = Temporal_Dynamics)) +
  geom_bar(stat = "identity", fill = "darkgrey", colour = "black", alpha = 0.7,
           width = 0.6)+
  scale_y_continuous(labels = scales::percent) + 
  theme_bw() + 
  labs(title ="Extreme Burn Severity", 
       x ="Temporal Dynamics", 
       y ="Relative Influence (%)") +
  theme(plot.title = element_text(hjust = 0.5, size = 10))+ 
  scale_x_discrete(expand = expansion(add = c(0.5, 0.5)))+
  theme(plot.margin = margin(2,2,2,2, "mm"))

ten_per_plot = ggarrange(rel.per.imten , 
                         rel.per.iqten , 
                         labels = c("A", "B"),
                         ncol = 2, 
                         nrow = 1,
                         widths = c(1,1),
                         common.legend = TRUE, 
                         legend="bottom")

ten_plot = annotate_figure(ten_per_plot, top = text_grob("Relative influence of Top 10 Predictors", 
                                                         color = "Black", face = "bold", size = 14))

#all
# median
rel.per.imall  = med_imp %>%
  mutate(Feature = 
           fct_reorder(Feature, Gain)) %>%
  mutate(Climate_Metrics = case_when(
    grepl("C", Feature) ~ "Climate Moisture Index",
    grepl("R", Feature) ~ "Relative Humidity",
    grepl("T", Feature) ~"Maximum Temperature"
  )) %>% 
  mutate(Temporal_Dynamics = case_when(
    grepl("yCsum", Feature) ~ "Yearly",
    grepl("yRmean", Feature) ~ "Yearly",
    grepl("mCsum", Feature) ~ "Monthly",
    grepl("mRmean", Feature) ~ "Monthly",
    grepl('m', Feature) ~ "Monthly",
    grepl('y', Feature) ~ "Yearly"
  )) %>% 
  group_by(Temporal_Dynamics) %>% 
  summarise(total_inf = sum(Gain)) %>% 
  ggplot(aes(y= total_inf, x = Temporal_Dynamics)) +
  geom_bar(stat = "identity", fill = c("darkgrey", "black"), colour = "black", alpha = 0.7,
           width = 0.6)+
  scale_y_continuous(labels = scales::percent) + 
  theme_bw() + 
  labs(title ="Median Burn Severity", 
       x ="Temporal Lag", 
       y ="Relative Importance (%)") +
  theme(plot.title = element_text(hjust = 0.5, size = 20),
        axis.title = element_text(size = 15),
        axis.text = element_text(size = 10))+ 
  scale_x_discrete(expand = expansion(add = c(0.5, 0.5)))+
  theme(plot.margin = margin(2,2,2,2, "mm"))

# Extreme
rel.per.iqall= ext_imp %>%
  mutate(Feature = 
           fct_reorder(Feature, Gain)) %>% 
  mutate(Climate_Metrics = case_when(
    grepl("C", Feature) ~ "Climate Moisture Index",
    grepl("R", Feature) ~ "Relative Humidity",
    grepl("T", Feature) ~"Maximum Temperature"
  )) %>% 
  mutate(Temporal_Dynamics = case_when(
    grepl("yCsum", Feature) ~ "Yearly",
    grepl("yRmean", Feature) ~ "Yearly",
    grepl("mCsum", Feature) ~ "Monthly",
    grepl("mRmean", Feature) ~ "Monthly",
    grepl('m', Feature) ~ "Monthly",
    grepl('y', Feature) ~ "Yearly"
  )) %>% 
  group_by(Temporal_Dynamics) %>% 
  summarise(total_inf = sum(Gain)) %>% 
  ggplot(aes(y= total_inf, x = Temporal_Dynamics)) +
  geom_bar(stat = "identity", fill = c("darkgrey", "black"), colour = "black", alpha = 0.7,
           width = 0.6)+
  scale_y_continuous(labels = scales::percent) + 
  theme_bw() +  
  labs(title ="Extreme Burn Severity", 
       x ="Temporal Lag", 
       y ="Relative Importance (%)") +
  theme(plot.title = element_text(hjust = 0.5, size = 20),
        axis.title = element_text(size = 15),
        axis.text = element_text(size = 10))+ 
  scale_x_discrete(expand = expansion(add = c(0.5, 0.5)))+
  theme(plot.margin = margin(2,2,2,2, "mm"))


all_per_plot = ggarrange(rel.per.imall ,
                         rel.per.iqall,
                         labels = c("C", "D"),
                         ncol = 2, 
                         nrow = 1,
                         widths = c(1,1),
                         common.legend = TRUE, legend="bottom")

all_plot = annotate_figure(all_per_plot,
                           top = text_grob("Relative influence of All Predictors", 
                           color = "Black", face = "bold", 
                           size = 14, vjust = 0.1))


ggarrange(ten_plot, all_plot,
          ncol =1, nrow = 2) + 
  theme(plot.margin = margin(0.1,0.1,2,0.1, "cm"))

# Figure 5 Percent Importance Predictors --------
rel.perm.mten = med_imp %>%
  mutate(Feature = 
           fct_reorder(Feature, Gain)) %>% 
  dplyr::slice_max(Gain, n = 10) %>% 
  mutate(Climate_Metrics = case_when(
    grepl("C", Feature) ~ "Climate Moisture Index",
    grepl("R", Feature) ~ "Relative Humidity",
    grepl("T", Feature) ~"Maximum Temperature"
  )) %>% 
  group_by(Climate_Metrics) %>% 
  summarise(total_inf = sum(Gain)) %>% 
  ggplot(aes(y= total_inf, x = Climate_Metrics)) +
  geom_bar(stat = "identity", fill = "darkgrey", colour = "black", alpha = 0.7,
           width = 0.6)+
  scale_y_continuous(labels = scales::percent) + 
  theme_bw() + 
  labs(title ="Median Burn Severity", 
       x ="Climate Metrics", 
       y ="Relative Influence (%)") +
  theme(plot.title = element_text(hjust = 0.5, size = 10))+ 
  scale_x_discrete(expand = expansion(add = c(0.5, 0.5)))+
  theme(plot.margin = margin(2,2,2,2, "mm"))
rel.perm.mten

rel.perm.qten = ext_imp %>%
  mutate(Feature = 
           fct_reorder(Feature, Gain)) %>% 
  dplyr::slice_max(Gain, n = 10) %>% 
  mutate(Climate_Metrics = case_when(
    grepl("C", Feature) ~ "Climate Moisture Index",
    grepl("R", Feature) ~ "Relative Humidity",
    grepl("T", Feature) ~"Maximum Temperature"
  )) %>% 
  group_by(Climate_Metrics) %>% 
  summarise(total_inf = sum(Gain)) %>% 
  ggplot(aes(y= total_inf, x = Climate_Metrics)) +
  geom_bar(stat = "identity", fill = "darkgrey", colour = "black", alpha = 0.7,
           width = 0.6)+
  scale_y_continuous(labels = scales::percent) + 
  theme_bw() + 
  labs(title ="Extreme Burn Severity", 
       x ="Climate Metrics", 
       y ="Relative Influence (%)") +
  theme(plot.title = element_text(hjust = 0.5, size = 10))+ 
  scale_x_discrete(expand = expansion(add = c(0.5, 0.5)))+
  theme(plot.margin = margin(2,2,2,2, "mm"))


ten_perm_plot = ggarrange(rel.perm.mten ,
                          rel.perm.qten , 
                          labels = c("A", "B"),
                          ncol = 2, 
                          nrow = 1,
                          widths = c(1,1),
                          common.legend = TRUE, 
                          legend="bottom")

tenm_plot = annotate_figure(ten_perm_plot, top = text_grob("Relative influence of Top 10 Predictors", 
                                                           color = "Black", face = "bold", size = 14))

# all
rel.perm.mall = med_imp %>%
  mutate(Feature = 
           fct_reorder(Feature, Gain)) %>% 
  mutate(Climate_Metrics = case_when(
    grepl("C", Feature) ~ "Climate Moisture Index",
    grepl("R", Feature) ~ "Relative Humidity",
    grepl("T", Feature) ~"Maximum Temperature"
  )) %>% 
  group_by(Climate_Metrics) %>% 
  summarise(total_inf = sum(Gain)) %>% 
  ggplot(aes(y= total_inf, x = Climate_Metrics)) +
  geom_bar(stat = "identity", fill = "darkgrey", colour = "black", alpha = 0.7,
           width = 0.6)+
  scale_y_continuous(labels = scales::percent) + 
  theme_bw() + 
  labs(title ="Median Burn Severity", 
       x ="Climate Metrics", 
       y ="Relative Influence (%)") +
  theme(plot.title = element_text(hjust = 0.5, size = 10))+ 
  scale_x_discrete(expand = expansion(add = c(0.5, 0.5)))+
  theme(plot.margin = margin(2,2,2,2, "mm"))


rel.perm.qall = ext_imp %>%
  mutate(Feature = 
           fct_reorder(Feature, Gain)) %>%  
  mutate(Climate_Metrics = case_when(
    grepl("C", Feature) ~ "Climate Moisture Index",
    grepl("R", Feature) ~ "Relative Humidity",
    grepl("T", Feature) ~"Maximum Temperature"
  )) %>% 
  group_by(Climate_Metrics) %>% 
  summarise(total_inf = sum(Gain)) %>% 
  ggplot(aes(y= total_inf, x = Climate_Metrics)) +
  geom_bar(stat = "identity", fill = "darkgrey", colour = "black", alpha = 0.7,
           width = 0.6)+
  scale_y_continuous(labels = scales::percent) + 
  theme_bw() + 
  labs(title ="Extreme Burn Severity", 
       x ="Climate Metrics", 
       y ="Relative Influence (%)") +
  theme(plot.title = element_text(hjust = 0.5, size = 10))+ 
  scale_x_discrete(expand = expansion(add = c(0.5, 0.5)))+
  theme(plot.margin = margin(2,2,2,2, "mm"))


all_perm_plot = ggarrange(rel.perm.mall ,  
                          rel.perm.qall , 
                          labels = c("C", "D"),
                          ncol = 2, 
                          nrow = 1,
                          widths = c(1,1),
                          common.legend = TRUE, 
                          legend="bottom")

all_m_plot = annotate_figure(all_perm_plot, top = text_grob("Relative influence of All Predictors", 
                                                            color = "Black", face = "bold", size = 14))


ggarrange(tenm_plot, all_m_plot,
          ncol =1, nrow = 2) + 
  theme(plot.margin = margin(0.1,0.1,2,0.1, "cm"))

# Figure 6 Percent Importance Cumulative --------

# Median
rel.perm.tall = med_imp %>%
  mutate(Feature = 
           fct_reorder(Feature, Gain)) %>% 
  mutate(Temporal_Cumulative_Effects = case_when(
    grepl("yCsum", Feature) ~ "Interannual",
    grepl("yRmean", Feature) ~ "Interannual",
    grepl("mCsum", Feature) ~ "Intra-annual",
    grepl("mRmean", Feature) ~"Intra-annual",
  ))  %>% 
  group_by(Temporal_Cumulative_Effects) %>%
  filter(!is.na(Temporal_Cumulative_Effects)) %>% 
  summarise(total_inf = sum(Gain)) %>% 
  ggplot(aes(y= total_inf, x = Temporal_Cumulative_Effects)) +
  geom_bar(stat = "identity", fill = "darkgrey", colour = "black", alpha = 0.7,
           width = 0.6)+
  scale_y_continuous(labels = scales::percent) + 
  theme_bw() + 
  labs(title ="Median Burn Severity", 
       x ="Temporal Cumulative Effects", 
       y ="Relative Influence (%)") +
  theme(plot.title = element_text(hjust = 0.5, size = 10))+ 
  scale_x_discrete(expand = expansion(add = c(0.5, 0.5)))+
  theme(plot.margin = margin(2,2,2,2, "mm"))
rel.perm.tall

# Extreme
rel.perq.tall = ext_imp %>%
  mutate(Feature = 
           fct_reorder(Feature, Gain)) %>%  
  mutate(Temporal_Cumulative_Effects = case_when(
    grepl("yCsum", Feature) ~ "Interannual",
    grepl("yRmean", Feature) ~ "Interannual",
    grepl("mCsum", Feature) ~ "Intra-annual",
    grepl("mRmean", Feature) ~"Intra-annual",
  ))  %>% 
  group_by(Temporal_Cumulative_Effects) %>%
  filter(!is.na(Temporal_Cumulative_Effects)) %>% 
  summarise(total_inf = sum(Gain)) %>% 
  ggplot(aes(y= total_inf, x =Temporal_Cumulative_Effects)) +
  geom_bar(stat = "identity", fill = "darkgrey", colour = "black", alpha = 0.7,
           width = 0.6)+
  scale_y_continuous(labels = scales::percent) + 
  theme_bw() + 
  labs(title ="Extreme Burn Severity", 
       x ="Temporal Cumulative Effects", 
       y ="Relative Influence (%)") +
  theme(plot.title = element_text(hjust = 0.5, size = 10))+ 
  scale_x_discrete(expand = expansion(add = c(0.5, 0.5)))+
  theme(plot.margin = margin(2,2,2,2, "mm"))



all_pert_plot = ggarrange(rel.perm.tall ,  
                          rel.perq.tall , 
                          labels = c("C", "D"),
                          ncol = 2, 
                          nrow = 1,
                          widths = c(1,1),
                          common.legend = TRUE, 
                          legend="bottom")

all_t_plot = annotate_figure(all_pert_plot, top = text_grob("Relative influence Among Cumulative Metrics", 
                                                            color = "Black", face = "bold", size = 14))
### temp vs non-cumulative

#Median
rel.perm.tall_ny = med_imp %>%
  mutate(Feature = 
           fct_reorder(Feature, Gain)) %>% 
  mutate(Temporal_Cumulative_Effects = case_when(
    grepl("yCsum", Feature) ~ "Cumulative",
    grepl("yRmean", Feature) ~ "Cumulative",
    grepl("mCsum", Feature) ~ "Cumulative",
    grepl("mRmean", Feature) ~"Cumulative",
  ))  %>% 
  group_by(Temporal_Cumulative_Effects) %>%
  replace(is.na(.), "Non-Cumulative") %>% 
  summarise(total_inf = sum(Gain)) %>% 
  ggplot(aes(y= total_inf, x = Temporal_Cumulative_Effects)) +
  geom_bar(stat = "identity", fill = "darkgrey", colour = "black", alpha = 0.7,
           width = 0.6)+
  scale_y_continuous(labels = scales::percent) + 
  theme_bw() + 
  labs(title ="Median Burn Severity", 
       x ="Temporal Cumulative Effects", 
       y ="Relative Influence (%)") +
  theme(plot.title = element_text(hjust = 0.5, size = 10))+ 
  scale_x_discrete(expand = expansion(add = c(0.5, 0.5)))+
  theme(plot.margin = margin(2,2,2,2, "mm"))
rel.perm.tall_ny

#Extreme

rel.perq.tall_ny = ext_imp %>%
  mutate(Feature = 
           fct_reorder(Feature, Gain)) %>%  
  mutate(Temporal_Cumulative_Effects = case_when(
    grepl("yCsum", Feature) ~ "Cumulative",
    grepl("yRmean", Feature) ~ "Cumulative",
    grepl("mCsum", Feature) ~ "Cumulative",
    grepl("mRmean", Feature) ~"Cumulative",
  ))  %>% 
  group_by(Temporal_Cumulative_Effects) %>%
  replace(is.na(.), "Non-Cumulative") %>% 
  summarise(total_inf = sum(Gain)) %>% 
  ggplot(aes(y= total_inf, x =Temporal_Cumulative_Effects)) +
  geom_bar(stat = "identity", fill = "darkgrey", colour = "black", alpha = 0.7,
           width = 0.6)+
  scale_y_continuous(labels = scales::percent) + 
  theme_bw() + 
  labs(title ="Extreme Burn Severity", 
       x ="Temporal Cumulative Effects", 
       y ="Relative Influence (%)") +
  theme(plot.title = element_text(hjust = 0.5, size = 10))+ 
  scale_x_discrete(expand = expansion(add = c(0.5, 0.5)))+
  theme(plot.margin = margin(2,2,2,2, "mm"))



all_pert_plot_ny = ggarrange(rel.perm.tall_ny ,  
                             rel.perq.tall_ny , 
                             labels = c("A", "B"),
                             ncol = 2, 
                             nrow = 1,
                             widths = c(1,1),
                             common.legend = TRUE, 
                             legend="bottom")

all_t_plot_ny = annotate_figure(all_pert_plot_ny, top = text_grob("Relative influence of All Predictors", 
                                                                  color = "Black", face = "bold", size = 14))


ggarrange(all_t_plot_ny, all_t_plot,
          ncol =1, nrow = 2) + 
  theme(plot.margin = margin(0.1,0.1,2,0.1, "cm"))

# figure S1 full model relative importance -----------
# Median
full.imp.im.plot= med_imp %>%
  mutate(Feature = fct_reorder(Feature, Gain)) %>% 
  mutate(Climate_Metrics = case_when(
    grepl("C", Feature) ~ "Climate Moisture Index",
    grepl("R", Feature) ~ "Relative Humidity",
    grepl("T", Feature) ~"Maximum Temperature"
  )) %>% 
  mutate(Temporal_Dynamics = case_when(
    grepl("yCsum", Feature) ~ "Yearly",
    grepl("yRmean", Feature) ~ "Yearly",
    grepl("mCsum", Feature) ~ "Monthly",
    grepl("mRmean", Feature) ~ "Monthly",
    grepl('m', Feature) ~ "Monthly",
    grepl('y', Feature) ~ "Yearly"
  )) %>% 
  ggplot(aes(Gain, Feature, fill = Climate_Metrics, width = .8), color = "black") +
  geom_col_pattern(
    aes(pattern = Temporal_Dynamics),
    colour = "black",
    pattern_fill = "black",
    pattern_angle = 45,
    pattern_density = 0.05,
    pattern_spacing = 0.03,
    position = position_dodge2(preserve = 'single')) +
  coord_flip() +
  scale_pattern_manual(values = c("none", "stripe"), "Temporal Dynamics",
                       guide = guide_legend(override.aes = list(fill = "white"))) +
  theme(text = element_text(size = 10)) +
  
  scale_fill_manual(values = c("#808080", "#C0C0C0", "#313335"), "Climate Metrics", 
                    guide = guide_legend(override.aes = list(pattern = "none")))+
  theme_bw()+
  labs(title = "Median Burn Severity",  x = "Relative Influence (%)", y = "Climate Predictor") +
  scale_x_continuous(labels = scales::percent)+
  theme(plot.title = element_text(hjust = 0.5))+
  theme(panel.background = element_blank()) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))


# Extremes
full.imp.iq.plot= ext_imp %>%
  mutate(Feature = fct_reorder(Feature, Gain)) %>% 
  mutate(Climate_Metrics = case_when(
    grepl("C", Feature) ~ "Climate Moisture Index",
    grepl("R", Feature) ~ "Relative Humidity",
    grepl("T", Feature) ~"Maximum Temperature"
  )) %>% 
  mutate(Temporal_Dynamics = case_when(
    grepl("yCsum", Feature) ~ "Yearly",
    grepl("yRmean", Feature) ~ "Yearly",
    grepl("mCsum", Feature) ~ "Monthly",
    grepl("mRmean", Feature) ~ "Monthly",
    grepl('m', Feature) ~ "Monthly",
    grepl('y', Feature) ~ "Yearly"
  )) %>% 
  ggplot(
    aes(Gain, Feature, fill = Climate_Metrics, width = .8), color = "black") +
  geom_col_pattern(
    aes(pattern = Temporal_Dynamics),
    colour = "black",
    pattern_fill = "black",
    pattern_angle = 45,
    pattern_density = 0.05,
    pattern_spacing = 0.015,
    position = position_dodge2(preserve = 'single')) +
  coord_flip() +
  scale_pattern_manual(values = c("none", "stripe"), "Temporal Dynamics",
                       guide = guide_legend(override.aes = list(fill = "white"))) +
  theme(text = element_text(size = 10)) +
  
  scale_fill_manual(values = c("#808080", "#C0C0C0", "#313335"), "Climate Metrics", 
                    guide = guide_legend(override.aes = list(pattern = "none")))+
  scale_x_continuous(labels = scales::percent)+
  theme_bw()+
  labs(title = "Extreme Burn Severity",  x = "Relative Influence (%)", y = "Climate Predictor") +
  theme(plot.title = element_text(hjust = 0.5))+
  theme(panel.background = element_blank()) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

full.imp.iq.plot

# arrange plots
ggarrange(full.imp.iq.plot, full.imp.im.plot, 
          labels = c("A", "B"),
          ncol = 1, nrow = 2, widths = c(1,1),
          heights = c(5,5),
          common.legend = TRUE, legend="bottom")

# Figure S2 model performance -----------------

# median
med_tib_training = med_res %>%  collect_metrics(summarize = FALSE)  
med_tib_training = med_tib_training %>%  mutate(dataset = c("training", "training","training", "training", "training", "training", "training", "training", "training", "training",
                                                    "training", "training","training", "training","training", "training","training", "training","training", "training"))
med_tib_test = medtest_res %>%  collect_metrics(summarize = FALSE)
med_tib_test = med_tib_test %>%  mutate(dataset = c("testing", "testing", "testing", "testing", "testing", "testing", "testing", "testing", "testing", "testing", "testing", "testing", "testing", "testing",
                                            "testing", "testing", "testing", "testing", "testing", "testing"))

xlabels = c("Testing", "Training")
perf_tib_med = med_tib_training %>%  full_join(med_tib_test)  %>% 
  pivot_wider( names_from = ".metric",
               values_from = ".estimate") 
perf_plotmed_rmse = perf_tib_med %>%  
  ggplot(aes(x = dataset, y = rmse, fill = dataset)) +
  geom_boxplot(fill = "darkgrey", alpha=0.7) +
  stat_summary(fun=mean, geom="point", shape=20, size=8, color="black", fill="black") +
  theme(legend.position="none") +
  theme_bw() + 
  theme(legend.position="none") +
  labs(title = "Median Burn Severity", y= "RMSE") +
  theme(plot.title = element_text(hjust = 0.5)) +
  scale_x_discrete(labels = xlabels)+
  theme(axis.title.x = element_blank(), 
        axis.text = element_text(size = 10),
        plot.title = element_text(size =15))


perf_plotmed_rsq = perf_tib_med %>%  
  ggplot(aes(x = dataset, y = rsq, fill = dataset)) +
  geom_boxplot(fill= "darkgrey", alpha=0.7) +
  stat_summary(fun=mean, geom="point", shape=20, size=8, color="black", fill="black") +
  theme(legend.position="none") + 
  theme_bw() + 
  theme(legend.position="none") +
  labs(title = "Median Burn Severity",  y= "R-squared") +
  theme(plot.title = element_text(hjust = 0.5))+
  theme(axis.title.x = element_blank()) +
  scale_x_discrete(labels = xlabels)+
  theme(axis.title.x = element_blank(), 
        axis.text = element_text(size = 10),
        plot.title = element_text(size =15)) 


# Extreme
ext_tib_train = ext_res %>%  collect_metrics(summarize = FALSE)  
ext_tib_train = ext_tib_train %>%  mutate(dataset = c("training", "training","training", "training", "training", "training", "training", "training", "training", "training",
                                          "training", "training","training", "training","training", "training","training", "training","training", "training"))
ext_tib_test = exttest_res %>%  collect_metrics(summarize = FALSE)
ext_tib_test = ext_tib_test %>%  mutate(dataset = c("testing", "testing", "testing", "testing", "testing", "testing", "testing", "testing", "testing", "testing", "testing", "testing", "testing", "testing",
                                        "testing", "testing", "testing", "testing", "testing", "testing"))

xlabels = c("Testing", "Training")
perf_tib_ext = ext_tib_train %>%  full_join(ext_tib_test)  %>% 
  pivot_wider( names_from = ".metric",
               values_from = ".estimate") 
plot_perfext_rmse = perf_tib_ext %>%  
  ggplot(aes(x = dataset, y = rmse, fill = dataset)) +
  geom_boxplot(fill = "darkgrey", alpha=0.7) +
  stat_summary(fun=mean, geom="point", shape=20, size=8, color="black", fill="black") +
  theme(legend.position="none") +
  theme_bw() + 
  theme(legend.position="none") +
  labs(title = "Extreme Burn Severity", y= "RMSE") +
  theme(plot.title = element_text(hjust = 0.5)) +
  scale_x_discrete(labels = xlabels)+
  theme(axis.title.x = element_blank(), 
        axis.text = element_text(size = 10),
        plot.title = element_text(size =15))


plot_perfext_rsq= perf_tib_ext %>% 
  ggplot(aes(x = dataset, y = rsq, fill = dataset)) +
  geom_boxplot(fill= "darkgrey", alpha=0.7) +
  stat_summary(fun=mean, geom="point", shape=20, size=8, color="black", fill="black") +
  theme(legend.position="none") + 
  theme_bw() + 
  theme(legend.position="none") +
  labs(title = "Extreme Burn Severity",  y= "R-squared") +
  theme(plot.title = element_text(hjust = 0.5))+
  theme(axis.title.x = element_blank()) +
  scale_x_discrete(labels = xlabels)+
  theme(axis.title.x = element_blank(), 
        axis.text = element_text(size = 10),
        plot.title = element_text(size =15)) 

# arrange plots in a grid
perf_plot = ggarrange(perf_plotmed_rmse, perf_plotmed_rsq, 
                      plot_perfext_rmse, plot_perfext_rsq, 
                      labels = c("A", "B", "C", "D"),
                      ncol = 2, nrow = 2, widths = c(1,1),
                      common.legend = TRUE, legend="bottom")

# add title
annotate_figure(perf_plot, top = text_grob("Model Validation and Prediction Accuracy", 
                                           color = "Black", face = "bold", size = 14))

# Figure S3 Extreme Partial dependence plots -----

ext_train = ext_train %>% mutate_if(is.integer,as.numeric)

# create container
mario_explainer <- explain_tidymodels(
  ext_fit,
  data = dplyr::select(ext_train, -RBR_quant),
  y = as.integer(ext_train$RBR_quant),
  verbose = FALSE
)

#plots per predictor
pdp_time <- model_profile(
  mario_explainer,
  variables = "8mR",
  N = NULL,
  variable_splits = list("8mR"=unique(ext_train$`8mR`))
)

rh_iq_8 <-as_tibble(pdp_time$agr_profiles) %>%
  mutate(`_label_` = str_remove(`_label_`, "workflow_")) %>%
  ggplot(aes(`_x_`, `_yhat_`,)) +
  geom_line(size = 1.2, alpha = 0.8) +
  theme_bw()+
  labs(
    x = "Relative Humidity (%)",
    y = "Extreme Burn Severity",
    color = NULL,
    title = "8 Month Time-Lag",
  ) +
  theme(plot.title = element_text(hjust = 0.5))+
  theme(plot.subtitle = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 10)) 
rh_iq_8

#9mR
ninemR <- model_profile(
  mario_explainer,
  variables = c("9mR"),
  N = NULL,
  variable_splits = list("9mR"=unique(ext_train$`9mR`))
)

rh_iq_9 <- as_tibble(ninemR$agr_profiles) %>%
  mutate(`_label_` = str_remove(`_label_`, "workflow_")) %>%
  ggplot(aes(`_x_`, `_yhat_`,)) +
  geom_line(size = 1.2, alpha = 0.8) +
  theme_bw()+
  labs(
    x = "Relative Humidity (%)",
    y = "Extreme Burn Severity",
    color = NULL,
    title = "9 Month Time-Lag"
  ) +
  theme(plot.title = element_text(hjust = 0.5))+
  theme(plot.subtitle = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 10)) 
rh_iq_9



#7mR
sevenr <- model_profile(
  mario_explainer,
  variables = c("7mR"),
  N = NULL,
  variable_splits = list("7mR"=unique(ext_train$`7mR`))
  
)

r_iq_7m <- as_tibble(sevenr$agr_profiles) %>%
  mutate(`_label_` = str_remove(`_label_`, "workflow_")) %>%
  ggplot(aes(`_x_`, `_yhat_`,)) +
  geom_line(size = 1.2, alpha = 0.8) +
  theme_bw()+
  labs(
    x = "Relative Humidity (%)",
    y = "Extreme Burn Severity",
    color = NULL,
    title = "7 Month Time-Lag"
  ) +
  theme(plot.title = element_text(hjust = 0.5))+
  theme(plot.subtitle = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 10)) 
r_iq_7m

#4yR
foury <- model_profile(
  mario_explainer,
  variables = c("4yR"),
  N = NULL,
  variable_splits = list("4yR"=unique(ext_train$`4yR`))
  
)

r_iq_4y <- as_tibble(foury$agr_profiles) %>%
  mutate(`_label_` = str_remove(`_label_`, "workflow_")) %>%
  ggplot(aes(`_x_`, `_yhat_`,)) +
  geom_line(size = 1.2, alpha = 0.8) +
  theme_bw()+
  labs(
    x = "Relative Humidity (%)",
    y = "Extreme Burn Severity",
    color = NULL,
    title = "4 Year Time-Lag"
  ) +
  theme(plot.title = element_text(hjust = 0.5))+
  theme(plot.subtitle = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 10)) 
r_iq_4y

#7mC
sevenC <- model_profile(
  mario_explainer,
  variables = c("7mC"),
  N = NULL,
  variable_splits = list("7mC"=unique(ext_train$`7mC`))
  
)

c_iq_7m <- as_tibble(sevenC$agr_profiles) %>%
  mutate(`_label_` = str_remove(`_label_`, "workflow_")) %>%
  ggplot(aes(`_x_`, `_yhat_`,)) +
  geom_line(size = 1.2, alpha = 0.8) +
  theme_bw()+
  labs(
    x = "Climate Mosituer index (mm)",
    y = "Extreme Burn Severity",
    color = NULL,
    title = "7 Month Time-Lag"
  ) +
  theme(plot.title = element_text(hjust = 0.5))+
  theme(plot.subtitle = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 10)) 
c_iq_7m

#3mR
threer <- model_profile(
  mario_explainer,
  variables = c("3mR"),
  N = NULL,
  variable_splits = list("3mR"=unique(ext_train$`3mR`))
  
)

r_iq_3m <- as_tibble(threer$agr_profiles) %>%
  mutate(`_label_` = str_remove(`_label_`, "workflow_")) %>%
  ggplot(aes(`_x_`, `_yhat_`,)) +
  geom_line(size = 1.2, alpha = 0.8) +
  theme_bw()+
  labs(
    x = "Relative Humidity (%)",
    y = "Extreme Burn Severity",
    color = NULL,
    title = "3 Month Time-Lag"
  ) +
  theme(plot.title = element_text(hjust = 0.5))+
  theme(plot.subtitle = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 10)) 
r_iq_3m


#4yC

fouryc <- model_profile(
  mario_explainer,
  variables = c("4yC"),
  N = NULL,
  variable_splits = list("4yC"=unique(ext_train$`4yC`))
  
)

c_iq_4y <- as_tibble(fouryc$agr_profiles) %>%
  mutate(`_label_` = str_remove(`_label_`, "workflow_")) %>%
  ggplot(aes(`_x_`, `_yhat_`,)) +
  geom_line(size = 1.2, alpha = 0.8) +
  theme_bw()+
  labs(
    x = "Relative Humidity (%)",
    y = "Extreme Burn Severity",
    color = NULL,
    title = "4 Year Time-Lag"
  ) +
  theme(plot.title = element_text(hjust = 0.5))+
  theme(plot.subtitle = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 10)) 
c_iq_4y


#8mC
eightmC <- model_profile(
  mario_explainer,
  variables = c("8mC"),
  N = NULL,
  variable_splits = list("8mC"=unique(ext_train$`8mC`))
  
)

c_iq_8m <- as_tibble(eightmC$agr_profiles) %>%
  mutate(`_label_` = str_remove(`_label_`, "workflow_")) %>%
  ggplot(aes(`_x_`, `_yhat_`,)) +
  geom_line(size = 1.2, alpha = 0.8) +
  theme_bw()+
  labs(
    x = "Climate Mosituer index (mm)",
    y = "Extreme Burn Severity",
    color = NULL,
    title = "8 Month Time-Lag"
  ) +
  theme(plot.title = element_text(hjust = 0.5))+
  theme(plot.subtitle = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 10)) 
c_iq_8m

#6mC
sixmC <- model_profile(
  mario_explainer,
  variables = c("6mC"),
  N = NULL,
  variable_splits = list("6mC"=unique(ext_train$`6mC`))
  
)

c_iq_6m <- as_tibble(sixmC$agr_profiles) %>%
  mutate(`_label_` = str_remove(`_label_`, "workflow_")) %>%
  ggplot(aes(`_x_`, `_yhat_`,)) +
  geom_line(size = 1.2, alpha = 0.8) +
  theme_bw()+
  labs(
    x = "Climate Mosituer index (mm)",
    y = "Extreme Burn Severity",
    color = NULL,
    title = "6 Month Time-Lag"
  ) +
  theme(plot.title = element_text(hjust = 0.5))+
  theme(plot.subtitle = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 10)) 
c_iq_6m

#7mC
nineCsum <- model_profile(
  mario_explainer,
  variables = c("9mCsum"),
  N = NULL,
  variable_splits = list("9mCsum"=unique(ext_train$`9mCsum`))
  
)

csum_iq_9m <- as_tibble(nineCsum$agr_profiles) %>%
  mutate(`_label_` = str_remove(`_label_`, "workflow_")) %>%
  ggplot(aes(`_x_`, `_yhat_`,)) +
  geom_line(size = 1.2, alpha = 0.8) +
  theme_bw()+
  labs(
    x = "Climate Mosituer index (mm)",
    y = "Extreme Burn Severity",
    color = NULL,
    title = "9 Month Cumulative"
  ) +
  theme(plot.title = element_text(hjust = 0.5))+
  theme(plot.subtitle = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 10)) 
csum_iq_9m

### arrange grid
pdp_10 = ggarrange(rh_iq_8, rh_iq_9, r_iq_4y, c_iq_7m, r_iq_3m,
                   c_iq_4y, c_iq_8m, c_iq_6m, r_iq_7m, csum_iq_9m,
                   labels = c("1", "2", "3", "4", "5", "6",
                              "7", "8", "9", "10"),
                   vjust = 3,
                   ncol = 5, nrow = 2, widths = c(3,3))

annotate_figure(pdp_10, 
                top = text_grob("Partial Dependence of 10 Most Influential Predicts of Extreme Burn Severity", 
                                color = "Black", 
                                face = "bold", 
                                size = 14,
                                vjust = 0.2))

# Figure S4 Median Partial dependence plots ------
med_train = med_train %>% mutate_if(is.integer,as.numeric)

#3mR
mario_explainer2 <- explain_tidymodels(
  med_fit,
  data = dplyr::select(med_train, -RBR_median),
  y = as.integer(med_train$RBR_median),
  verbose = FALSE
)


pdp_time <- model_profile(
  mario_explainer,
  variables = c("3mR"),
  N = NULL,
  variable_splits = list("3mR"=unique(med_train$`3mR`))
)

mr3 = as_tibble(pdp_time$agr_profiles) %>%
  mutate(`_label_` = str_remove(`_label_`, "workflow_")) %>%
  ggplot(aes(`_x_`, `_yhat_`,)) +
  geom_line(size = 1.2, alpha = 0.8) +
  theme_bw()+
  labs(
    x = " Relative Humidity (%)",
    y = "Median Burn Severity",
    color = NULL,
    title = "3 Month Time-Lag"
  ) +
  theme(plot.title = element_text(hjust = 0.5))+
  theme(plot.subtitle = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 10)) 
mr3

#6mC
sixC <- model_profile(
  mario_explainer,
  variables = c("6mC"),
  N = NULL,
  variable_splits = list("6mC"=unique(med_train$`6mC`))
)

mc6 = as_tibble(sixC$agr_profiles) %>%
  mutate(`_label_` = str_remove(`_label_`, "workflow_")) %>%
  ggplot(aes(`_x_`, `_yhat_`,)) +
  geom_line(size = 1.2, alpha = 0.8) +
  theme_bw()+
  labs(
    x = "Climate Moisture Index (mm)",
    y = "Median Burn Severity",
    color = NULL,
    title = "6 Month Time-Lag"
  ) +
  theme(plot.title = element_text(hjust = 0.5))+
  theme(plot.subtitle = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 10)) 
mc6


#3yC
threeyC <- model_profile(
  mario_explainer,
  variables = c("3yC"),
  N = NULL,
  variable_splits = list("3yC"=unique(med_train$`3yC`))
)

yc3 = as_tibble(threeyC$agr_profiles) %>%
  mutate(`_label_` = str_remove(`_label_`, "workflow_")) %>%
  ggplot(aes(`_x_`, `_yhat_`,)) +
  geom_line(size = 1.2, alpha = 0.8) +
  theme_bw()+
  labs(
    x = "Climate Moisture Index (mm)",
    y = "Median Burn Severity",
    color = NULL,
    title = "3 Year Time-Lag"
  ) +
  theme(plot.title = element_text(hjust = 0.5))+
  theme(plot.subtitle = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 10)) 
yc3


#6mCsum
sixmCs <- model_profile(
  mario_explainer,
  variables = c("6mCsum"),
  N = NULL,
  variable_splits = list("6mCsum"=unique(med_train$`6mCsum`))
)

sixmCs_m = as_tibble(sixmCs$agr_profiles) %>%
  mutate(`_label_` = str_remove(`_label_`, "workflow_")) %>%
  ggplot(aes(`_x_`, `_yhat_`,)) +
  geom_line(size = 1.2, alpha = 0.8) +
  theme_bw()+
  labs(
    x = "Climate Moisture Index (mm)",
    y = "Median Burn Severity",
    color = NULL,
    title = "6 Month Cumumlative"
  ) +
  theme(plot.title = element_text(hjust = 0.5))+
  theme(plot.subtitle = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 10)) 
sixmCs_m
#8mR
mr8 <- model_profile(
  mario_explainer,
  variables = c("8mR"),
  N = NULL,
  variable_splits = list("8mR"=unique(med_train$`8mR`))
)

mr8_m = as_tibble(mr8$agr_profiles) %>%
  mutate(`_label_` = str_remove(`_label_`, "workflow_")) %>%
  ggplot(aes(`_x_`, `_yhat_`,)) +
  geom_line(size = 1.2, alpha = 0.8) +
  theme_bw()+
  labs(
    x = "Relative Humidity (%)",
    y = "Median Burn Severity",
    color = NULL,
    title = "8 Month Time-Lag"
  ) +
  theme(plot.title = element_text(hjust = 0.5))+
  theme(plot.subtitle = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 10)) 
mr8_m
#12mR
mr12 <- model_profile(
  mario_explainer,
  variables = c("12mR"),
  N = NULL,
  variable_splits = list("12mR"=unique(med_train$`12mR`))
)

mr12_m = as_tibble(mr12$agr_profiles) %>%
  mutate(`_label_` = str_remove(`_label_`, "workflow_")) %>%
  ggplot(aes(`_x_`, `_yhat_`,)) +
  geom_line(size = 1.2, alpha = 0.8) +
  theme_bw()+
  labs(
    x = "Relative Humidity (%)",
    y = "Median Burn Severity",
    color = NULL,
    title = "12 Month Time-Lag"
  ) +
  theme(plot.title = element_text(hjust = 0.5))+
  theme(plot.subtitle = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 10)) 
mr12_m
#9mC
mc9 <- model_profile(
  mario_explainer,
  variables = c("9mC"),
  N = NULL,
  variable_splits = list("9mC"=unique(med_train$`9mC`))
)

mc9_m = as_tibble(mc9$agr_profiles) %>%
  mutate(`_label_` = str_remove(`_label_`, "workflow_")) %>%
  ggplot(aes(`_x_`, `_yhat_`,)) +
  geom_line(size = 1.2, alpha = 0.8) +
  theme_bw()+
  labs(
    x = "Climate Moisture Index (mm)",
    y = "Median Burn Severity",
    color = NULL,
    title = "9 Month Time-Lag"
  ) +
  theme(plot.title = element_text(hjust = 0.5))+
  theme(plot.subtitle = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 10)) 
mc9_m
#4yR
ry4 <- model_profile(
  mario_explainer,
  variables = c("4yR"),
  N = NULL,
  variable_splits = list("4yR"=unique(med_train$`4yR`))
)

fourr_y = as_tibble(ry4$agr_profiles) %>%
  mutate(`_label_` = str_remove(`_label_`, "workflow_")) %>%
  ggplot(aes(`_x_`, `_yhat_`,)) +
  geom_line(size = 1.2, alpha = 0.8) +
  theme_bw()+
  labs(
    x = "Relative Humidity (%)",
    y = "Median Burn Severity",
    color = NULL,
    title = "4 Year Time-Lag"
  ) +
  theme(plot.title = element_text(hjust = 0.5))+
  theme(plot.subtitle = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 10)) 
fourr_y
#0yC
cy0 <- model_profile(
  mario_explainer,
  variables = c("0yC"),
  N = NULL,
  variable_splits = list("0yC"=unique(med_train$`0yC`))
)

zeroc_y = as_tibble(cy0$agr_profiles) %>%
  mutate(`_label_` = str_remove(`_label_`, "workflow_")) %>%
  ggplot(aes(`_x_`, `_yhat_`,)) +
  geom_line(size = 1.2, alpha = 0.8) +
  theme_bw()+
  labs(
    x = "Climate Moisture Index (mm)",
    y = "Median Burn Severity",
    color = NULL,
    title = "12 Month Cumulative"
  ) +
  theme(plot.title = element_text(hjust = 0.5))+
  theme(plot.subtitle = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 10)) 
zeroc_y
#2mC
mc2 <- model_profile(
  mario_explainer,
  variables = c("2mC"),
  N = NULL,
  variable_splits = list("2mC"=unique(med_train$`2mC`))
)

c2_m = as_tibble(mc2$agr_profiles) %>%
  mutate(`_label_` = str_remove(`_label_`, "workflow_")) %>%
  ggplot(aes(`_x_`, `_yhat_`,)) +
  geom_line(size = 1.2, alpha = 0.8) +
  theme_bw()+
  labs(
    x = "Climate Moisture Index (mm)",
    y = "Median Burn Severity",
    color = NULL,
    title = "2 Month Time-Lag"
  ) +
  theme(plot.title = element_text(hjust = 0.5))+
  theme(plot.subtitle = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 10)) 
c2_m

pdp_10_m = ggarrange(mr3, mc6, yc3, sixmCs_m, mr8_m,
                     mr12_m, mc9_m, fourr_y, zeroc_y, c2_m,
                     labels = c("1", "2", "3", "4", "5", "6",
                                "7", "8", "9", "10"),
                     vjust = 3,
                     ncol = 5, nrow = 2, widths = c(3,3))

annotate_figure(pdp_10_m, 
                top = text_grob("Partial Dependence of 10 Most Influential Predicts for Median Burn Severity", 
                                color = "Black", 
                                face = "bold", 
                                size = 14,
                                vjust = 0.2))
