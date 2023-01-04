### Stochastic Gradient Boosted Regression Tree Analysis ###


#required packages
library(caret)
library(gbm)
library(dplyr)
library(tidyverse)
library(ggplot2)
library(tidymodels)
library(readr)
library(gridExtra)


# Data preparation ---------------------

### DATA ###

data_tlm <- read_csv2("data/TimeLagMoisture_data.csv")

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
cvFolds_exttest <- ext_test %>% vfold_cv(10)
exttest_fit <- ext_wf %>% 
  finalize_workflow(select_best(ext_tune)) %>% 
  fit(ext_test)
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
cvFolds_medtest <- med_test %>% vfold_cv(10)
medtest_fit <- med_wf %>% 
  finalize_workflow(select_best(med_tune)) %>% 
  fit(med_test)

medtest_res <-  medtest_fit %>% fit_resamples(resamples = cvFolds_medtest, control = keep_pred)

medtest_res %>%  collect_metrics() # testing
med_res %>%  collect_metrics(summarize = TRUE)# training


# model performance -----------------

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
perf_plot = ggarrange(perf_plotm_rmse, perf_plotmed_rsq, 
                      plot_perfext_rmse, plot_perfext_rsq, 
                      labels = c("A", "B", "C", "D"),
                      ncol = 2, nrow = 2, widths = c(1,1),
                      common.legend = TRUE, legend="bottom")

# add title
annotate_figure(perf_plot, top = text_grob("Model Validation and Prediction Accuracy", 
                                           color = "Black", face = "bold", size = 14))
