library(dplyr)
library(psych)
library(magrittr)
library(tidyverse)
library(tidymodels)
library(conflicted)
library(fastshap)
library(vip)
library(recipeselectors)
library(bonsai) 
library(lightgbm)
library(caret)
library(shapviz)

conflicts_prefer(dplyr::select)
conflicts_prefer(dplyr::filter)
conflicts_prefer(yardstick::recall)
conflicts_prefer(yardstick::precision)

###### General population sample 

general_pop_sample <- read.csv("general_pop_sample.csv")

# Partitioning
set.seed(4435)
general_pop_split_data <- initial_split(
  data = general_pop_sample,
  prop = 0.75,
  strata = "ue_ever_unreal_voice"
)

# Inspect how many individuals are voice hearers and non-voice hearers in order to determine how many partitions to create

general_pop_training_data <- training(general_pop_split_data)
general_pop_testing_data <- testing(general_pop_split_data)

general_pop_data_0 <- general_pop_training_data %>% filter(ue_ever_unreal_voice == 0)
general_pop_data_1 <- general_pop_training_data %>% filter(ue_ever_unreal_voice == 1)

general_pop_n_partitions <- ceiling(nrow(general_pop_data_0) / nrow(general_pop_data_1))

general_pop_partitions <- vector("list", general_pop_n_partitions)

general_pop_n_1 <- nrow(general_pop_data_1)

set.seed(54)
general_pop_indices <- sample(seq_len(nrow(general_pop_data_0)))

for (i in seq_len(general_pop_n_partitions)) {
  
  # Calculate indices for the current subset of the negative class
  start_idx <- (i - 1) * general_pop_n_1 + 1
  end_idx <- min(i * general_pop_n_1, nrow(general_pop_data_0))  # Ensure the last subset isn't too large
  
  # Subset the negative class
  general_pop_subsample_0 <- general_pop_data_0[general_pop_indices[start_idx:end_idx], ]
  
  # For the last partition, reduce the positive class to match the size of the remaining negative class subset
  if (nrow(general_pop_subsample_0) < general_pop_n_1) {
    general_pop_subsample_1 <- general_pop_data_1[sample(nrow(general_pop_data_1), nrow(general_pop_subsample_0)), ]
  } else {
    general_pop_subsample_1 <- general_pop_data_1
  }
  
  # Combine the current subset of the negative class with the positive class
  general_pop_partitions[[i]] <- bind_rows(general_pop_subsample_0, general_pop_subsample_1)
}

# Exactly balanced bagging for 73 models

model_times <- list()

for (i in 1:73){
  
  
  general_pop_partitions_training_data <- general_pop_partitions[[i]]
  
  
  model_recipe <- recipes::recipe(
    ue_ever_unreal_voice ~ ., data = general_pop_partitions_training_data) %>%
    update_role(eid, new_role = "ID") %>%
    update_role(age, new_role = "age") %>%
    step_zv(all_predictors()) %>%
    step_nzv(all_predictors()) %>%
    step_range(all_predictors(), -all_factor(),-all_nominal(), min = 0, max = 1) %>%
    step_unknown(all_nominal_predictors(), all_factor_predictors()) %>%  # Handle unknowns for all categorical variables
    step_dummy(all_nominal_predictors(), all_factor_predictors(), one_hot = TRUE) %>%
    step_zv(all_predictors()) %>%
    step_nzv(all_predictors())
  
  # cross validaiton  
  cv_folds <- vfold_cv(general_pop_partitions_training_data, strata = ue_ever_unreal_voice, repeats = 10)
  
  # Create a model and workflow
  xgb_model <- boost_tree(
    trees = tune(),
    learn_rate = tune(),
    tree_depth = tune(),
    min_n = tune(),
    loss_reduction = tune(),
    sample_size = tune(),
    mtry = tune()
  ) %>%
    set_mode("classification") %>%
    set_engine("xgboost")
  
  workflow.xgboost <- workflow() %>%
    add_recipe(model_recipe) %>%
    add_model(xgb_model)
  
  xgboost_params <- parameters(
    trees(range = c(500, 1500)),
    learn_rate(),
    tree_depth(),
    min_n(),
    loss_reduction()
    sample_size = sample_prop(),
    finalize(mtry(), general_pop_partitions_training_data)
  )
  
  set.seed(321)
  # Start timing
  start_time <- Sys.time()
  xgboost_tune <- workflow.xgboost %>%
    tune_bayes(
      resamples = cv_folds,
      param_info = xgboost_params,
      iter = 100,
      metrics = metric_set(brier_class),
      control = control_bayes(no_improve = 25, save_pred = TRUE, verbose = TRUE),
      initial = 10
    )
  
  end_time <- Sys.time()
  elapsed_time <- end_time - start_time
  
  model_times[[i]] <- elapsed_time
  
  # Return results
  general_pop_model_results <- list(
    model_recipe = model_recipe,
    cv_folds = cv_folds,
    xgb_model = xgb_model,
    workflow.xgboost = workflow.xgboost,
    xgboost_params = xgboost_params,
    xgboost_tune = xgboost_tune,
    best_brier_class = select_best(xgboost_tune, metric = "brier_class")
  )
  
  saveRDS(general_pop_model_results, 
          file = paste0("D:\\training_partitions\\general_pop_model_results_", i, ".rds"))
  
  
  print(paste("Saved all objects for model", i))
  print(paste("Time taken for model", i, ":", elapsed_time))
}

saveRDS(model_times, file = "general_pop_model_times.rds")

# Exactly balanced bagging: Perform ensemble modeling using bagging

all_model.pred_1 <- list()
all_model.pred_0 <- list()
all_model.pred_class <- list()

for (i in 1:73){
  
  general_pop_training_data_sets <- general_pop_partitions[[i]]
  
  file_path.general_pop <- paste0("D:\\training_partitions\\general_pop_model_results_", i, ".rds")
  
  gen_pop_output.general_pop <- readRDS(file_path.general_pop)
  
  final_xgb <- finalize_workflow(gen_pop_output.general_pop$workflow.xgboost, gen_pop_output.general_pop$best_brier_class)
  
  fitted_workflow <- final_xgb %>%
    fit(data = gen_pop_training_data_sets)
  
  model_predictions <- predict(fitted_workflow, new_data = gen_pop_testing_data, type = "prob")
  
  all_model.pred_1[[i]] <- model_predictions$.pred_1
  all_model.pred_0[[i]] <- model_predictions$.pred_0
  
  class_predictions <- predict(fitted_workflow, new_data = gen_pop_testing_data, type = "class")
  
  all_model.pred_class[[i]] <- class_predictions$.pred_class
  print(paste("Model", i, "done"))
}

write.csv(all_model.pred_1, "gen_pop_prob_1.rds", row.names = FALSE)
write.csv(all_model.pred_0, "gen_pop_prob_0.rds", row.names = FALSE)
write.csv(all_model.pred_class, "gen_pop_class.rds", row.names = FALSE)

# Step 1: Combine the predictions into a data frame
pred_1_df <- as.data.frame(all_model.pred_1)
pred_0_df <- as.data.frame(all_model.pred_0)
pred_class_df <- as.data.frame(all_model.pred_class)

# Step 2: Calculate the average probabilities for class 1 and class 0
avg_pred_1 <- rowMeans(pred_1_df)
avg_pred_0 <- rowMeans(pred_0_df)

majority_vote <- function(x) {
  # Return the most frequent class (mode)
  return(names(sort(table(x), decreasing = TRUE))[1])
}

majority_voted_predictions <- apply(pred_class_df, 1, majority_vote)

majority_voted_predictions <- factor(majority_voted_predictions, levels = c("0", "1"))

# Step 3: Create a data frame with the new columns to be added
new_columns <- data.frame(
  .avg_pred_1 = avg_pred_1,
  .avg_pred_0 = avg_pred_0,
  .pred_class = majority_voted_predictions
)

gen_pop_test_bagging <- as_tibble(cbind(new_columns, gen_pop_testing_data))

# save
write.csv(gen_pop_test_bagging, "gen_pop_test_bagged.csv", row.names = FALSE)

# Exactly balanced bagging: Assess the models

gen_pop_test_bagging$.pred_class  <- as.factor(gen_pop_test_bagging$.pred_class )
gen_pop_test_bagging$ue_ever_unreal_voice  <- as.factor(gen_pop_test_bagging$ue_ever_unreal_voice )

gen_pop_test_bagging %>%
  bal_accuracy(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")

gen_pop_test_bagging %>%
  roc_auc(truth = ue_ever_unreal_voice, .avg_pred_1, event_level = "second")


brier_class(
  data = gen_pop_test_bagging,
  truth = ue_ever_unreal_voice,
  .avg_pred_0)


gen_pop_test_bagging %>%
  pr_auc(truth = ue_ever_unreal_voice, 
         .avg_pred_1, 
         event_level = "second")

gen_pop_test_bagging %>%
  accuracy(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")


gen_pop_test_bagging %>%
  f_meas(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")


gen_pop_test_bagging %>%
  mcc(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")

gen_pop_test_bagging %>%
  precision(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")

gen_pop_test_bagging %>%
  recall(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")

# General population sample: Roc curve

gen_pop_roc_curve <- gen_pop_test_bagging %>%
  roc_curve(truth = ue_ever_unreal_voice, .avg_pred_1, event_level = "second") %>%
  autoplot() +
  theme_minimal() +
  theme(
    panel.grid.major = element_blank(),  
    panel.grid.minor = element_blank(),  
    axis.line = element_line(),          
    axis.ticks = element_line()          
  ) +
  coord_fixed(expand = FALSE)

saveRDS(gen_pop_roc_curve, "gen_pop_roc_curve.rds")

# General population sample: Metrics for all the individual models

file_path_gen_pop <- "D:\\training_partitions\\"


for (i in 1:73){
  
  file_name_gen_pop <- paste0(file_path_gen_pop, "general_pop_model_results_", i, ".rds")
  
  gen_pop_model_results <- readRDS(file_name_gen_pop)
  
  training_data <- gen_pop_partitions[[i]]
  
  
  final_xgb <- finalize_workflow(gen_pop_model_results$workflow.xgboost, gen_pop_model_results$best_precision)
  
  # Fit the finalized workflow on the training data
  fitted_workflow <- final_xgb %>%
    fit(data = training_data)
  
  # Recipe Specification
  recipe_spec <- fitted_workflow %>%
    extract_recipe()
  recipe_spec
  
  # Extract the recipe from the fitted workflow (if needed)
  extracted_recipe <- fitted_workflow %>%
    extract_recipe()
  
  print(extracted_recipe)
  
  # Make predictions on the test data
  xgb.pred <- predict(fitted_workflow, new_data = gen_pop_testing_data, type = "prob")
  print(xgb.pred)
  
  # Assess the Model
  
  #attach predictions to training dataset
  training_results <- augment(fitted_workflow, training_data)  
  
  training_results %>%
    roc_curve(truth = ue_ever_unreal_voice, .pred_1, event_level = "second")  %>%
    autoplot()
  
  #attach predictions to testing dataset
  testing_results <- augment(fitted_workflow, gen_pop_testing_data)
  
  testing_results %>%
    roc_curve(truth = ue_ever_unreal_voice, .pred_1, event_level = "second")  %>%
    autoplot()
  
  # Calculate and print ROC AUC for training and test sets
  train_auc <- training_results %>%
    roc_auc(truth = ue_ever_unreal_voice, .pred_1,  event_level = "second")
  print(train_auc)
  
  test_auc <- testing_results %>%
    roc_auc(truth = ue_ever_unreal_voice, .pred_1, event_level = "second")
  print(test_auc)
  
  # Calculate and print Brier scores
  train_brier <- brier_class(
    data = training_results,
    truth = ue_ever_unreal_voice,
    .pred_1,
  )
  print(train_brier)
  
  test_brier <- brier_class(
    data = testing_results,
    truth = ue_ever_unreal_voice,
    .pred_1,
  )
  print(test_brier)
  
  # Calculate and print a_PRC 
  
  a_prc_train <- training_results %>%
    pr_auc(truth = ue_ever_unreal_voice, 
           .pred_1, 
           event_level = "second")
  print(a_prc_train)
  
  a_prc_test <- testing_results %>%
    pr_auc(truth = ue_ever_unreal_voice, 
           .pred_1, 
           event_level = "second")
  print(a_prc_test)
  
  # Calculate and print balanced accuracy
  train_bal_acc <- training_results %>% 
    bal_accuracy(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
  print(train_bal_acc)
  
  test_bal_acc <- testing_results %>%
    bal_accuracy(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
  print(test_bal_acc)
  
  # Calculate and print F1
  train_f1<- training_results %>% 
    f_meas(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
  print(train_f1)
  
  test_f1 <- testing_results %>%
    f_meas(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
  print(test_f1)
  
  # Calculate and print balanced accuracy
  train_mcc<- training_results %>% 
    mcc(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
  print(train_mcc)
  
  test_mcc <- testing_results %>%
    mcc(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
  print(test_mcc)
  
  # Calculate and print precision
  train_precision<- training_results %>% 
    precision(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
  print(train_precision)
  
  test_precision <- testing_results %>%
    precision(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
  print(test_precision)
  
  # Calculate and print precision
  train_recall<- training_results %>% 
    recall(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
  print(train_recall)
  
  test_recall <- testing_results %>%
    recall(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
  print(test_recall)
  
  train_specificity<- training_results %>% 
    specificity(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
  print(train_specificity)
  
  test_specificity <- testing_results %>%
    specificity(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
  print(test_specificity)
  
  train_npv<- training_results %>% 
    npv(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
  print(train_npv)
  
  test_npv <- testing_results %>%
    npv(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
  print(test_npv)
  
  
  # Create data frames for metrics
  gen_pop_train_metrics <- tibble(
    Train_AUC = train_auc$.estimate,
    Train_Accuracy = train_accuracy,
    Train_Brier_Score = train_brier,
    Train_a_PRC = a_prc_train,
    Train_bal_acc = train_bal_acc,
    Train_F_meas = train_f1,
    Train_mcc = train_mcc,
    Train_precision = train_precision,
    Train_recall = train_recall,
    train_specificity,
    train_npv
  )
  
  gen_pop_test_metrics <- tibble(
    Test_AUC = test_auc$.estimate,
    Test_Accuracy = test_accuracy,
    Test_Brier_Score = test_brier,
    Test_a_PRC = a_prc_test,
    Test_bal_acc = test_bal_acc,
    Test_F_meas = test_f1,
    Test_mcc = test_mcc,
    Test_precision = test_precision,
    Test_recall = test_recall,
    test_specificity,
    test_npv
  )
}

# General population sample: Performance per iteration

for (i in 1:73) {
  
  training_data <- gen_pop_partitions[[i]]
  
  file_path.gen_pop <- paste0("D:\\training_partitions\\general_pop_model_results_", i, ".rds")
  
  gen_pop_output <- readRDS(file_path.gen_pop)
  
  all_train_metrics <- list()  
  all_test_metrics <- list()
  
  for (fold_index in seq_len(nrow(gen_pop_output$cv_iteration))) {
    
    repetition_num <- gen_pop_output$cv_iteration$id[iteration_index]  
    fold_num <- gen_pop_output$cv_iteration$id2[iteration_index]        
    
    split <- gen_pop_output$cv_iteration$splits[[iteration_index]]
    training_iteration <- analysis(split)
    testing_iteration <- assessment(split)
    
    final_xgb <- finalize_workflow(gen_pop_output$workflow.xgboost, gen_pop_output$best_brier_class)
    
    fitted_workflow <- final_xgb %>%
      fit(data = training_iteration)
    
    xgb_training_pred <- 
      predict(fitted_workflow, training_iteration) %>%  # Class predictions
      bind_cols(predict(fitted_workflow, training_iteration, type = "prob")) %>%  # Probabilities
      bind_cols(training_iteration %>% select(ue_ever_unreal_voice)) 
    
    # Calculate and store test predictions
    xgb_test_pred <- 
      predict(fitted_workflow, testing_iteration) %>% 
      bind_cols(predict(fitted_workflow, testing_iteration, type = "prob")) %>% 
      bind_cols(testing_iteration %>% select(ue_ever_unreal_voice))
    
    # Calculate and print ROC AUC for training and test sets
    train_auc <- xgb_training_pred %>%
      roc_auc(truth = ue_ever_unreal_voice, .pred_1, event_level = "second")
    
    test_auc <- xgb_test_pred %>%
      roc_auc(truth = ue_ever_unreal_voice, .pred_1, event_level = "second")
    
    # Calculate and print accuracy
    train_accuracy <- xgb_training_pred %>%
      accuracy(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
    
    test_accuracy <- xgb_test_pred %>%
      accuracy(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
    
    # Calculate and print Brier scores
    train_brier <- brier_class(
      data = xgb_training_pred,
      truth = ue_ever_unreal_voice,
      .pred_0
    )
    
    test_brier <- brier_class(
      data = xgb_test_pred,
      truth = ue_ever_unreal_voice,
      .pred_0
    )
    
    # Calculate and print a_PRC 
    a_prc_train <- xgb_training_pred %>%
      pr_auc(truth = ue_ever_unreal_voice, 
             .pred_1, 
             event_level = "second")
    
    a_prc_test <- xgb_test_pred %>%
      pr_auc(truth = ue_ever_unreal_voice, 
             .pred_1, 
             event_level = "second")
    
    # Calculate and print balanced accuracy
    train_bal_acc <- xgb_training_pred %>% 
      bal_accuracy(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
    
    test_bal_acc <- xgb_test_pred %>%
      bal_accuracy(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
    
    # Calculate and print F1
    train_f1<- xgb_training_pred %>% 
      f_meas(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
    
    test_f1 <- xgb_test_pred %>%
      f_meas(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
    
    # Calculate MCC
    train_mcc<- xgb_training_pred %>% 
      mcc(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
    
    test_mcc <- xgb_test_pred %>%
      mcc(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
    
    # Calculate precision
    train_precision<- xgb_training_pred %>% 
      precision(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
    
    test_precision <- xgb_test_pred %>%
      precision(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
    
    # Calculate recall
    train_recall<- xgb_training_pred %>% 
      recall(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
    
    test_recall<- xgb_test_pred %>% 
      recall(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
    
    # Calculate specificity
    train_spec<- xgb_training_pred %>% 
      specificity(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
    
    test_spec<- xgb_test_pred %>% 
      specificity(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
    
    # Store iteration metrics
    gen_pop_iteration_train_metrics <- tibble(
      Model = i,
      Repetition = as.integer(gsub("Repeat", "", repetition_num)),
      Fold = as.integer(gsub("Fold", "", fold_num)),
      Train_AUC = train_auc$.estimate,
      Train_accuracy = train_accuracy$.estimate,
      Train_bal_acc = train_bal_acc$.estimate,
      Train_Brier_Score = train_brier$.estimate,
      Train_a_PRC = a_prc_train$.estimate,
      Train_F_meas = train_f1$.estimate,
      Train_mcc = train_mcc$.estimate,
      Train_precision = train_precision$.estimate,
      Train_recall = train_recall$.estimate,
      Train_spec = train_spec$.estimate
    )
    
    gen_pop_iteration_test_metrics <- tibble(
      Model = i,
      Repetition = as.integer(gsub("Repeat", "", repetition_num)),
      Fold = as.integer(gsub("Fold", "", fold_num)),
      Test_AUC = test_auc$.estimate,
      Test_accuracy = test_accuracy$.estimate,
      Test_bal_acc = test_bal_acc$.estimate,
      Test_Brier_Score = test_brier$.estimate,
      Test_a_PRC = a_prc_test$.estimate,
      Test_F_meas = test_f1$.estimate,
      Test_mcc = test_mcc$.estimate,
      Test_precision = test_precision$.estimate,
      Test_recall = test_recall$.estimate,
      Test_spec = test_spec$.estimate
    )
    
    all_train_metrics[[paste(repetition_num, fold_num, sep = "_")]] <- train_metrics
    all_test_metrics[[paste(repetition_num, fold_num, sep = "_")]] <- test_metrics
    
    message(paste("Fold", fold_num, "in Repetition", repetition_num, "completed for model", i))
  }
  
  # Combine metrics across all folds
  train_metrics_table <- bind_rows(all_train_metrics)
  test_metrics_table <- bind_rows(all_test_metrics)
  
  train_summary <- train_metrics_table %>%
    group_by(Model, Repetition) %>%
    summarise(
      Train_M_AUC = mean(Train_AUC),
      Train_SD_AUC = sd(Train_AUC),
      Train_M_accuracy = mean(Train_accuracy),
      Train_SD_accuracy = sd(Train_accuracy),
      Train_M_bal_acc = mean(Train_bal_acc),
      Train_SD_bal_acc = sd(Train_bal_acc),
      Train_M_Brier_Score = mean(Train_Brier_Score),
      Train_SD_Brier_Score = sd(Train_Brier_Score),
      Train_M_a_PRC = mean(Train_a_PRC),
      Train_SD_a_PRC = sd(Train_a_PRC),
      Train_M_F_meas = mean(Train_F_meas),
      Train_SD_F_meas = sd(Train_F_meas),
      Train_M_mcc = mean(Train_mcc),
      Train_SD_mcc = sd(Train_mcc),
      Train_M_precision = mean(Train_precision),
      Train_SD_precision = sd(Train_precision),
      Train_M_recall = mean(Train_recall),
      Train_SD_recall = sd(Train_recall),
      Train_M_spec = mean(Train_spec),
      Train_SD_spec = sd(Train_spec),
      .groups = "drop"  # Drop grouping after summarization
    )
  
  test_summary <- test_metrics_table %>%
    group_by(Model, Repetition) %>%
    summarise(
      Test_M_AUC = mean(Test_AUC),
      Test_SD_AUC = sd(Test_AUC),
      Test_M_accuracy = mean(Test_accuracy),
      Test_SD_accuracy = sd(Test_accuracy),
      Test_M_bal_acc = mean(Test_bal_acc),
      Test_SD_bal_acc = sd(Test_bal_acc),
      Test_M_Brier_Score = mean(Test_Brier_Score),
      Test_SD_Brier_Score = sd(Test_Brier_Score),
      Test_M_a_PRC = mean(Test_a_PRC),
      Test_SD_a_PRC = sd(Test_a_PRC),
      Test_M_F_meas = mean(Test_F_meas),
      Test_SD_F_meas = sd(Test_F_meas),
      Test_M_mcc = mean(Test_mcc),
      Test_SD_mcc = sd(Test_mcc),
      Test_M_precision = mean(Test_precision),
      Test_SD_precision = sd(Test_precision),
      Test_M_recall = mean(Test_recall),
      Test_SD_recall = sd(Test_recall),
      Test_M_spec = mean(Test_spec),
      Test_SD_spec = sd(Test_spec),
      .groups = "drop"  # Drop grouping after summarization
    )
  

# General population sample: Evaluation of all general population sample models 

library(correctR)
model_comparisons <- data.frame(Model_Comparison = character(), 
                                t_statistic = numeric(), 
                                p_value = numeric(), 
                                df = numeric(), 
                                stringsAsFactors = FALSE)

# List to hold data frames for each model

model_data_list <- list()

# Loop through each model (1 to 73) and read data

for (i in 1:73) {
  
  # Read the data for the current model
  
  model_data <- read.csv(paste0("gen_pop_model_", i, "_test_metrics.csv"))
  
  # Prepare the data for analysis
  model_data <- model_data %>%
    mutate(model = i) %>%
    rename(values = Test_bal_acc, k = Fold, r = Repetition) %>%
    select(model, values, k, r)
  
  # Add to the list
  model_data_list[[i]] <- model_data
}

# Combine all models' data into one data frame
combined_data <- bind_rows(model_data_list)

# Generate all combinations of models for t-tests
model_combinations <- combn(1:73, 2)

# Loop through each combination and perform t-tests
for (combo in 1:ncol(model_combinations)) {
  model_1 <- model_combinations[1, combo]
  model_2 <- model_combinations[2, combo]
  
  # Filter data for the two models being compared
  pair_data <- combined_data %>% filter(model %in% c(model_1, model_2))
  
  # Perform the t-test
  ttest_result <- repkfold_ttest(data = pair_data, 
                                 n1 = 2402, 
                                 n2 = 268, 
                                 k = 10, 
                                 r = 10, 
                                 tailed = "one", 
                                 greater = model_2)  # Change as per hypothesis
  
  # Calculate degrees of freedom
  df <- 10 - 1  # Degrees of freedom for 10 folds
  
  # Store the results in the comparison table
  model_comparisons <- rbind(model_comparisons, 
                             data.frame(Model_Comparison = paste("Model", model_1, "vs Model", model_2),
                                        t_statistic = ttest_result$statistic,
                                        p_value = ttest_result$p.value,
                                        df = df,
                                        stringsAsFactors = FALSE))
}

# Print the comparison results table
print(model_comparisons)

## how many models reached significant before multiple correction
significant_comparisons <- model_comparisons %>%
  filter(p_value < 0.05)

holm <- model_comparisons %>% mutate(Adjusted_p_value = p.adjust(p_value, method = "holm", n = 2628))

significant_holm <- holm %>%
  filter(Adjusted_p_value < 0.05)

print(holm)
print(significant_holm)


# General population sample: The information gain and SHAP values - Rank sum

shap_rank_sum <- read.csv("non_clinical_73_models_shap_sum_ranks.csv")
gain_rank_sum <- read.csv("gen_pop_full_variables_rank_sum_73_models.csv")

shap_rank_sum <- shap_rank_sum %>%
  rename(Rank_sumShap = Rank_sum)

gain_rank_sum <- gain_rank_sum %>%
  rename(Rank_sumGain = Rank_sum)

rank_sum <- shap_rank_sum %>% 
  left_join(gain_rank_sum %>% select(Feature, Rank_sumGain), by = "Feature")

rank_sum <- rank_sum %>% arrange(Rank_sumGain)  # Arrange from low to high for Gain

# Left bar chart (Gain values, ordered from low to high)
p1 <- ggplot(rank_sum) +
  geom_bar(aes(x = -Rank_sumGain, y = reorder(Feature, -Rank_sumGain)), 
           stat = "identity", fill = "skyblue", width = 0.6) +
  labs(x = "Rank sum (Gain)", y = NULL) +
  theme_minimal() +
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    panel.grid = element_blank()
  )

# Text-only plot for feature names (placed in the center)
p2 <- ggplot(rank_sum) +
  geom_text(aes(x = 0, y = reorder(Feature, -Rank_sumGain), label = Feature), 
            hjust = 0.5, color = "black", size = 3) +  
  labs(x = "Sum Rank (Gain)", y = NULL) +
  theme_minimal() +
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    panel.grid = element_blank()
  )

# Right bar chart (SHAP values, ordered correctly)
p3 <- ggplot(rank_sum) +
  geom_bar(aes(x = Rank_sumShap, y = reorder(Feature, -Rank_sumGain)), 
           stat = "identity", fill = "#FFA500", width = 0.6) +  # Orange color for SHAP
  labs(x = "Sum Rank (SHAP)", y = NULL) +
  theme_minimal() +
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    panel.grid = element_blank()
  )

# Combine plots
combined_plot <- plot_grid(
  p1, p2, p3, 
  ncol = 3
)

# Display the combined plot
combined_plot

######## Psychosis sample

## Partitioning
psychosis_sample <- read.csv("psychosis_sample.csv")

set.seed(4)
psychosis_split_data <- initial_split(
  data = psychosis_sample,
  prop = 0.75,
  strata = "ue_ever_unreal_voice"
)


psychosis_training_data <- training(psychosis_split_data)
psychosis_testing_data <- testing(psychosis_split_data)

## The model

model_recipe <- recipes::recipe(
  ue_ever_unreal_voice ~ ., data = psychosis_training_data) %>%
  update_role(eid, new_role = "ID") %>%
  update_role(age, new_role = "age") %>%
  step_zv(all_predictors()) %>%
  step_nzv(all_predictors()) %>% 
  step_range(all_predictors(), -all_factor(),-all_nominal(), min = 0, max = 1) %>%
  step_unknown(all_nominal_predictors(), all_factor_predictors()) %>%  # Handle unknowns for all categorical variables
  step_dummy(all_nominal_predictors(), all_factor_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors()) %>%
  step_nzv(all_predictors())

# 4. cross validaiton  
cv_folds <- vfold_cv(psychosis_training_data, strata = ue_ever_unreal_voice, repeats = 10)

# Create a model and workflow
xgb_model <- boost_tree(
  trees = tune(),
  learn_rate = tune(),
  tree_depth = tune(),
  min_n = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  mtry = tune()
) %>%
  set_mode("classification") %>%
  set_engine("xgboost")#, scale_pos_weight = tune(), eval_metric = "auc")

workflow.xgboost <- workflow() %>%
  add_recipe(model_recipe) %>%
  add_model(xgb_model)

xgboost_params <- parameters(
  trees(range = c(500, 1000)),
  learn_rate(),  
  tree_depth(),  
  min_n(), 
  loss_reduction(),
  #scale_pos_weight(range = c(0.40, 0.50), trans = NULL), 
  sample_size = sample_prop(),  
  finalize(mtry(), psychosis_training_data)
)

set.seed(321)
# Start timing
tictoc::tic()
xgboost_tune <- workflow.xgboost %>%
  tune_bayes(
    resamples = cv_folds,
    param_info = xgboost_params,
    iter = 100,
    metrics = metric_set(brier_class), #,bal_accuracy, pr_auc, precision, f_meas, recall, mcc, mn_log_loss),
    control = control_bayes(no_improve = 25, save_pred = TRUE, verbose = TRUE),
    initial = 25
  )
tictoc::toc()


# Return results
psychosis_model_results <- list(
  model_recipe = model_recipe,
  cv_folds = cv_folds,
  xgb_model = xgb_model,
  workflow.xgboost = workflow.xgboost,
  xgboost_params = xgboost_params,
  xgboost_tune = xgboost_tune,
  best_brier_class = select_best(xgboost_tune, metric = "brier_class")
)

## Psychosis sample: Metrics

final_xgb <- finalize_workflow(psychosis_model_results$workflow.xgboost, psychosis_model_results$best_brier_class)

# Fit the finalized workflow on the training data
fitted_workflow <- final_xgb %>%
  fit(data = psychosis_training_data)

# Variable Importance Plot
vip(extract_fit_parsnip(fitted_workflow), geom = "point", num_features = 40)

# Recipe Specification
recipe_spec <- fitted_workflow %>%
  extract_recipe()
recipe_spec

# Extract the recipe from the fitted workflow (if needed)
extracted_recipe <- fitted_workflow %>%
  extract_recipe()

print(extracted_recipe)

# Make predictions on the test data
xgb.pred <- predict(fitted_workflow, new_data = psychosis_testing_data)
print(xgb.pred)

# Assess the Model
# Augment the test data with predictions
set.seed(2)
augment_xgboost <- augment(fitted_workflow, new_data = psychosis_testing_data)


psychosis_roc_curve <- augment_xgboost %>%
  roc_curve(truth = ue_ever_unreal_voice, .pred_1, event_level = "second") %>%
  autoplot()
print(psychosis_roc_curve)

# Calculate and store training predictions
set.seed(56)
xgb_training_pred <- 
  predict(fitted_workflow, psychosis_training_data) %>% 
  bind_cols(predict(fitted_workflow, psychosis_training_data, type = "prob")) %>% 
  bind_cols(psychosis_training_data %>% select(ue_ever_unreal_voice))

# Calculate and store test predictions
xgb_test_pred <- 
  predict(fitted_workflow, psychosis_testing_data) %>% 
  bind_cols(predict(fitted_workflow, psychosis_testing_data, type = "prob")) %>% 
  bind_cols(psychosis_testing_data %>% select(ue_ever_unreal_voice))


# Calculate and print ROC AUC for training and test sets
train_auc <- xgb_training_pred %>%
  roc_auc(truth = ue_ever_unreal_voice, .pred_1, event_level = "second")
print(train_auc)

test_auc <- xgb_test_pred %>%
  roc_auc(truth = ue_ever_unreal_voice, .pred_1, event_level = "second")
print(test_auc)

# Calculate and print accuracy
train_accuracy <- sum(xgb_training_pred$.pred_class == xgb_training_pred$ue_ever_unreal_voice) / nrow(xgb_training_pred)
print(tibble(Metric = "Train accuracy", Value = train_accuracy))

test_accuracy <- sum(xgb_test_pred$.pred_class == xgb_test_pred$ue_ever_unreal_voice) / nrow(xgb_test_pred)
print(tibble(Metric = "Test accuracy", Value = test_accuracy))

# Calculate and print Brier scores
train_brier <- brier_class(
  data = xgb_training_pred,
  truth = ue_ever_unreal_voice,
  .pred_0
)
print(train_brier)

test_brier <- brier_class(
  data = xgb_test_pred,
  truth = ue_ever_unreal_voice,
  .pred_0
)
print(test_brier)

# Calculate and print a_PRC 

a_prc_train <- xgb_training_pred %>%
  pr_auc(truth = ue_ever_unreal_voice, 
         .pred_1, 
         event_level = "second")
print(a_prc_train)

a_prc_test <- xgb_test_pred %>%
  pr_auc(truth = ue_ever_unreal_voice, 
         .pred_1, 
         event_level = "second")
print(a_prc_test)

# Calculate and print balanced accuracy
train_bal_acc <- xgb_training_pred %>% 
  bal_accuracy(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
print(train_bal_acc)

test_bal_acc <- xgb_test_pred %>%
  bal_accuracy(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
print(test_bal_acc)

# Calculate and print F1
train_f1<- xgb_training_pred %>% 
  f_meas(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
print(train_f1)

test_f1 <- xgb_test_pred %>%
  f_meas(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
print(test_f1)

# Calculate and print balanced accuracy
train_mcc<- xgb_training_pred %>% 
  mcc(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
print(train_mcc)

test_mcc <- xgb_test_pred %>%
  mcc(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
print(test_mcc)

# Calculate and print precision
train_precision<- xgb_training_pred %>% 
  precision(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
print(train_precision)

test_precision <- xgb_test_pred %>%
  precision(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
print(test_precision)

# Calculate and print precision
train_recall<- xgb_training_pred %>% 
  recall(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
print(train_recall)

test_recall <- xgb_test_pred %>%
  recall(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")
print(test_recall)

train_specificity<- xgb_training_pred %>% 
  specificity(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")

test_specificity <- xgb_test_pred %>%
  specificity(truth = ue_ever_unreal_voice, .pred_class , event_level = "second", estimator = "binary")

# Create data frames for metrics
train_metrics <- tibble(
  Train_AUC = train_auc$.estimate,
  Train_Accuracy = train_accuracy,
  Train_Brier_Score = train_brier,
  Train_a_PRC = a_prc_train,
  Train_bal_acc = train_bal_acc,
  Train_F_meas = train_f1,
  Train_mcc = train_mcc,
  Train_precision = train_precision,
  Train_recall = train_recall,
  train_specificity = train_specificity
)

print(train_metrics)

test_metrics <- tibble(
  Test_AUC = test_auc$.estimate,
  Test_Accuracy = test_accuracy,
  Test_Brier_Score = test_brier$.estimate,
  Test_a_PRC = a_prc_test$.estimate,
  Test_bal_acc = test_bal_acc$.estimate,
  Test_F_meas = test_f1$.estimate,
  Test_mcc = test_mcc$.estimate,
  Test_precision = test_precision$.estimate,
  Test_recall = test_recall$.estimate,
  Test_spec = test_specificity$.estimate
)

print(test_metrics)

# Psychosis sample: Threshold adjustment

library(probably)

threshold_data <- xgb_test_pred %>%
  threshold_perf(ue_ever_unreal_voice, .pred_0, thresholds = seq(0.1, 1, by = 0.0025), metric_set(sensitivity, specificity, bal_accuracy, j_index))

threshold_data %>%
  filter(.threshold %in% c(0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 1))

threshold_data <- threshold_data %>%
  filter(.metric != "distance") %>%
  mutate(group = ifelse(.metric %in% c("specificity", "sensitivity", "bal_accuracy","j_index"), "1", "2"))


max_j_index_threshold <- threshold_data %>%
  filter(.metric == "j_index") %>%
  filter(.estimate == max(.estimate)) %>%
  pull(.threshold)


ggplot(threshold_data, aes(x = .threshold, y = .estimate, color = .metric, alpha = group)) +
  geom_line() +
  theme_minimal() +
  scale_color_viridis_d(end = 0.9) +
  scale_alpha_manual(values = c(.4, 1), guide = "none") +
  geom_vline(xintercept = max_j_index_threshold, alpha = .6, color = "grey30") +
  labs(
    x = "Threshold",
    y = "Metric Estimate",
  )


# Print the best threshold based on the J-Index
threshold_data %>%
  filter(.threshold == max_j_index_threshold)

# Print the best threshold based on the J-Index
best_threshold <- threshold_data %>%
  filter(.threshold == max_j_index_threshold)


hard_pred_0.5675 <- xgb_test_pred %>%
  mutate(
    .pred = make_two_class_pred(
      estimate = .pred_0,
      levels = levels(ue_ever_unreal_voice),
      threshold = 0.5675
    )
  ) %>%
  select(ue_ever_unreal_voice, contains(".pred"))

hard_pred_0.5675$.pred <- as.factor(as.character(hard_pred_0.5675$.pred))

best_threshold_conf_matrix <- hard_pred_0.5675 %>%
  count(.truth = ue_ever_unreal_voice, .pred)


hard_pred_0.5 <- xgb_test_pred %>%
  mutate(
    .pred = make_two_class_pred(
      estimate = .pred_0,
      levels = levels(ue_ever_unreal_voice),
      threshold = 0.5
    )
  ) %>%
  select(ue_ever_unreal_voice, contains(".pred"))

hard_pred_0.5$.pred <- as.factor(as.character(hard_pred_0.5$.pred))

threshold_0.5_conf_matrix <- hard_pred_0.5 %>%
  count(.truth = ue_ever_unreal_voice, .pred)

j_index(hard_pred_0.5, ue_ever_unreal_voice, .pred)
j_index(hard_pred_0.5675, ue_ever_unreal_voice, .pred)

paste0("Threshold at 0.5675")
print(best_threshold_conf_matrix)

#  paste0("Threshold at 0.5")
print(threshold_0.5_conf_matrix)

## ROC_CURVE
roc_curve(hard_pred_0.5, truth = ue_ever_unreal_voice, .pred_1, event_level = "second") %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity)) +
  geom_path() +
  geom_abline(lty = 3) +
  coord_equal() +
  theme_bw()

psychosis_roc_curve<- roc_curve(hard_pred_0.5675, truth = ue_ever_unreal_voice, .pred_1, event_level = "second") %>%
  autoplot() +
  theme_minimal() +
  theme(
    panel.grid.major = element_blank(),  
    panel.grid.minor = element_blank(), 
    axis.line = element_line(),          
    axis.ticks = element_line()          
  ) +
  coord_fixed(expand = FALSE)

psychosis_roc_curve

hard_pred_0.5675 <- xgb_test_pred %>%
  mutate(
    .pred = make_two_class_pred(
      estimate = .pred_0,
      levels = levels(ue_ever_unreal_voice),
      threshold = 0.5675
    )
  )
hard_pred_0.5675 %>%
  roc_curve(truth = ue_ever_unreal_voice, .pred_1, event_level = "second")  %>%
  autoplot()


# Calculate and print ROC AUC for training and test sets
test_auc_0.5675 <- hard_pred_0.5675 %>%
  roc_auc(truth = ue_ever_unreal_voice, .pred_1,  event_level = "second")
print(train_auc)


test_accuracy_0.5675 <- hard_pred_0.5675 %>%
  accuracy(truth = ue_ever_unreal_voice, estimate = .pred)

test_bal_accuracy_0.5675 <- hard_pred_0.5675 %>%
  bal_accuracy(truth = ue_ever_unreal_voice, estimate = .pred)

test_brier_0.5675 <- brier_class(
  data = hard_pred_0.5675,
  truth = ue_ever_unreal_voice,
  .pred_0,
)
print(test_brier)


a_prc_test_0.5675 <- hard_pred_0.5675 %>%
  pr_auc(truth = ue_ever_unreal_voice, 
         .pred_1, 
         event_level = "second")
print(a_prc_test)


# Calculate and print F1

test_f1_0.5675 <- hard_pred_0.5675 %>%
  f_meas(truth = ue_ever_unreal_voice, .pred , event_level = "second", estimator = "binary")
print(test_f1)

# Calculate and print balanced accuracy

test_mcc_0.5675 <- hard_pred_0.5675 %>%
  mcc(truth = ue_ever_unreal_voice, .pred, event_level = "second", estimator = "binary")
print(test_mcc)

# Calculate and print precision
test_precision_0.5675 <- hard_pred_0.5675 %>%
  precision(truth = ue_ever_unreal_voice, .pred, event_level = "second", estimator = "binary")
print(test_precision)

# Calculate and print precision

test_recall_0.5675 <- hard_pred_0.5675 %>%
  recall(truth = ue_ever_unreal_voice, .pred , event_level = "second", estimator = "binary")
print(test_recall)

test_spec_0.5675 <- hard_pred_0.5675 %>%
  specificity(truth = ue_ever_unreal_voice, .pred, event_level = "second", estimator = "binary")

test_metrics_0.5675 <- tibble(
  Test_AUC = test_auc_0.5675$.estimate,
  Test_accuracy = test_accuracy_0.5675$.estimate,
  Test_bal_acc = test_bal_accuracy_0.5675$.estimate,
  Test_Brier_Score = test_brier_0.5675$.estimate,
  Test_a_PRC = a_prc_test_0.5675$.estimate,
  Test_F_meas = test_f1_0.5675$.estimate,
  Test_mcc = test_mcc_0.5675$.estimate,
  Test_precision = test_precision_0.5675$.estimate,
  Test_recall = test_recall_0.5675$.estimate,
  Test_spec = test_spec_0.5675$.estimate
)


# Psychosis sample: Performance per fold
all_train_metrics <- list()  
all_test_metrics <- list()

# Process each fold in the cross-validation for the model
for (fold_index in seq_len(nrow(psychosis_model_results$cv_folds))) {
  
  # Extract fold and repetition information
  repetition_num <- psychosis_model_results$cv_folds$id[fold_index]   # repeat ID
  fold_num <- psychosis_model_results$cv_folds$id2[fold_index]  
  
  split <- psychosis_model_results$cv_folds$splits[[fold_index]]
  training_fold <- analysis(split)
  testing_fold <- assessment(split)
  
  # Finalize and fit the model
  final_xgb <- finalize_workflow(psychosis_model_results$workflow.xgboost, psychosis_model_results$best_brier_class)
  fitted_workflow <- final_xgb %>% fit(data = training_fold)
  
  # Generate predictions for training and test sets
  xgb_training_pred <- predict(fitted_workflow, training_fold) %>%  
    bind_cols(predict(fitted_workflow, training_fold, type = "prob")) %>%  
    bind_cols(training_fold %>% select(ue_ever_unreal_voice))
  
  xgb_test_pred <- predict(fitted_workflow, testing_fold) %>% 
    bind_cols(predict(fitted_workflow, testing_fold, type = "prob")) %>% 
    bind_cols(testing_fold %>% select(ue_ever_unreal_voice))
  
  # Calculate metrics for training and test sets
  train_auc <- xgb_training_pred %>% roc_auc(truth = ue_ever_unreal_voice, .pred_1, event_level = "second")
  test_auc <- xgb_test_pred %>% roc_auc(truth = ue_ever_unreal_voice, .pred_1, event_level = "second")
  
  train_accuracy <- xgb_training_pred %>% accuracy(truth = ue_ever_unreal_voice, .pred_class, event_level = "second", estimator = "binary")
  test_accuracy <- xgb_test_pred %>% accuracy(truth = ue_ever_unreal_voice, .pred_class, event_level = "second", estimator = "binary")
  
  train_brier <- brier_class(data = xgb_training_pred, truth = ue_ever_unreal_voice, .pred_0)
  test_brier <- brier_class(data = xgb_test_pred, truth = ue_ever_unreal_voice, .pred_0)
  
  a_prc_train <- xgb_training_pred %>% pr_auc(truth = ue_ever_unreal_voice, .pred_1, event_level = "second")
  a_prc_test <- xgb_test_pred %>% pr_auc(truth = ue_ever_unreal_voice, .pred_1, event_level = "second")
  
  train_bal_acc <- xgb_training_pred %>% bal_accuracy(truth = ue_ever_unreal_voice, .pred_class, event_level = "second", estimator = "binary")
  test_bal_acc <- xgb_test_pred %>% bal_accuracy(truth = ue_ever_unreal_voice, .pred_class, event_level = "second", estimator = "binary")
  
  train_f1 <- xgb_training_pred %>% f_meas(truth = ue_ever_unreal_voice, .pred_class, event_level = "second", estimator = "binary")
  test_f1 <- xgb_test_pred %>% f_meas(truth = ue_ever_unreal_voice, .pred_class, event_level = "second", estimator = "binary")
  
  train_mcc <- xgb_training_pred %>% mcc(truth = ue_ever_unreal_voice, .pred_class, event_level = "second", estimator = "binary")
  test_mcc <- xgb_test_pred %>% mcc(truth = ue_ever_unreal_voice, .pred_class, event_level = "second", estimator = "binary")
  
  train_precision <- xgb_training_pred %>% precision(truth = ue_ever_unreal_voice, .pred_class, event_level = "second", estimator = "binary")
  test_precision <- xgb_test_pred %>% precision(truth = ue_ever_unreal_voice, .pred_class, event_level = "second", estimator = "binary")
  
  train_recall <- xgb_training_pred %>% recall(truth = ue_ever_unreal_voice, .pred_class, event_level = "second", estimator = "binary")
  test_recall <- xgb_test_pred %>% recall(truth = ue_ever_unreal_voice, .pred_class, event_level = "second", estimator = "binary")
  
  train_spec <- xgb_training_pred %>% specificity(truth = ue_ever_unreal_voice, .pred_class, event_level = "second", estimator = "binary")
  test_spec <- xgb_test_pred %>% specificity(truth = ue_ever_unreal_voice, .pred_class, event_level = "second", estimator = "binary")
  
  # Store metrics for each fold
  train_metrics <- tibble(
    Model = 1,  # Directly specify model ID
    Repetition = as.integer(gsub("Repeat", "", repetition_num)),
    Fold = as.integer(gsub("Fold", "", fold_num)),
    Train_AUC = train_auc$.estimate,
    Train_accuracy = train_accuracy$.estimate,
    Train_bal_acc = train_bal_acc$.estimate,
    Train_Brier_Score = train_brier$.estimate,
    Train_a_PRC = a_prc_train$.estimate,
    Train_F_meas = train_f1$.estimate,
    Train_mcc = train_mcc$.estimate,
    Train_precision = train_precision$.estimate,
    Train_recall = train_recall$.estimate,
    Train_spec = train_spec$.estimate
  )
  
  
  
  test_metrics <- tibble(
    Model = 1,  # Directly specify model ID
    Repetition = as.integer(gsub("Repeat", "", repetition_num)),
    Fold = as.integer(gsub("Fold", "", fold_num)),
    Test_AUC = test_auc$.estimate,
    Test_accuracy = test_accuracy$.estimate,
    Test_bal_acc = test_bal_acc$.estimate,
    Test_Brier_Score = test_brier$.estimate,
    Test_a_PRC = a_prc_test$.estimate,
    Test_F_meas = test_f1$.estimate,
    Test_mcc = test_mcc$.estimate,
    Test_precision = test_precision$.estimate,
    Test_recall = test_recall$.estimate,
    Test_spec = test_spec$.estimate
  )
  
  
  all_train_metrics[[paste(repetition_num, fold_num, sep = "_")]] <- train_metrics
  all_test_metrics[[paste(repetition_num, fold_num, sep = "_")]] <- test_metrics
  
  message(paste("Fold", fold_num, "in Repetition", repetition_num, "completed for model 1"))
}

# Combine all fold metrics into data frames
train_metrics_table <- bind_rows(all_train_metrics)
test_metrics_table <- bind_rows(all_test_metrics)

# Print final tables for training and testing metrics
print(train_metrics_table)
print(test_metrics_table)

train_summary <- train_metrics_table %>%
  group_by(Model, Repetition) %>%
  summarise(
    Train_AUC_Mean = mean(Train_AUC),
    Train_AUC_SD = sd(Train_AUC),
    Train_accuracy_Mean = mean(Train_accuracy),
    Train_accuracy_SD = sd(Train_accuracy),
    Train_bal_acc_Mean = mean(Train_bal_acc),
    Train_bal_acc_SD = sd(Train_bal_acc),
    Train_Brier_Score_Mean = mean(Train_Brier_Score),
    Train_Brier_Score_SD = sd(Train_Brier_Score),
    Train_a_PRC_Mean = mean(Train_a_PRC),
    Train_a_PRC_SD = sd(Train_a_PRC),
    Train_F_meas_Mean = mean(Train_F_meas),
    Train_F_meas_SD = sd(Train_F_meas),
    Train_mcc_Mean = mean(Train_mcc),
    Train_mcc_SD = sd(Train_mcc),
    Train_precision_Mean = mean(Train_precision),
    Train_precision_SD = sd(Train_precision),
    Train_recall_Mean = mean(Train_recall),
    Train_recall_SD = sd(Train_recall),
    Train_spec_Mean = mean(Train_spec),
    Train_spec_SD = sd(Train_spec),
    .groups = "drop"  # Drop grouping after summarization
  )

test_summary <- test_metrics_table %>%
  group_by(Model, Repetition) %>%
  summarise(
    Test_AUC_Mean = mean(Test_AUC),
    Test_AUC_SD = sd(Test_AUC),
    Test_accuracy_Mean = mean(Test_accuracy),
    Test_accuracy_SD = sd(Test_accuracy),
    Test_bal_acc_Mean = mean(Test_bal_acc),
    Test_bal_acc_SD = sd(Test_bal_acc),
    Test_Brier_Score_Mean = mean(Test_Brier_Score),
    Test_Brier_Score_SD = sd(Test_Brier_Score),
    Test_a_PRC_Mean = mean(Test_a_PRC),
    Test_a_PRC_SD = sd(Test_a_PRC),
    Test_F_meas_Mean = mean(Test_F_meas),
    Test_F_meas_SD = sd(Test_F_meas),
    Test_mcc_Mean = mean(Test_mcc),
    Test_mcc_SD = sd(Test_mcc),
    Test_precision_Mean = mean(Test_precision),
    Test_precision_SD = sd(Test_precision),
    Test_recall_Mean = mean(Test_recall),
    Test_recall_SD = sd(Test_recall),
    Test_spec_Mean = mean(Test_spec),
    Test_spec_SD = sd(Test_spec),
    .groups = "drop"  # Drop grouping after summarization
  )

# Psychosis sample: The information gain and the SHAP values

psychosis_final_xgb <- finalize_workflow(psychosis_model_results_nov$workflow.xgboost, psychosis_model_results_nov$best_brier_class)

psychosis_fitted_workflow <- psychosis_final_xgb %>%
  fit(data = psychosis_model_train_nov)

psychosis_extract_fitted_workflow.train <- extract_fit_engine(psychosis_fitted_workflow)
psychosis_importance_scores.train <- xgb.importance(model=psychosis_extract_fitted_workflow.train)

xgb.plot.importance(psychosis_importance_scores.train)

# Perform SHAP analysis (train)
psychosis_xgboost_shapviz.train <- shapviz::shapviz(
  extract_fit_engine(psychosis_fitted_workflow), 
  X_pred = bake(
    prep(psychosis_model_results_nov$model_recipe),
    has_role("predictor"),
    new_data = NULL,
    composition = "matrix"), interactions = TRUE)

psychosis_xgboost_shapviz.train$S <- -psychosis_xgboost_shapviz.train$S



# Merge SHAP importance with Gain values
importance_data <- psychosis_importance_scores.train %>%
  inner_join(shap_importance, by = "Feature") %>%
  arrange(desc(SHAP_Importance)) %>%  # Order by SHAP importance
  slice(1:20)  # Select top 20 features

# Ensure Gain values and feature names match the SHAP order
importance_data$Feature <- factor(importance_data$Feature, levels = importance_data$Feature)

# SHAP Importance Plot (pp1)
pp1 <- shapviz::sv_importance(psychosis_xgboost_shapviz.train, 
                              kind = "beeswarm", 
                              max_display = 20, 
                              show_numbers = TRUE) +
  theme_minimal() +
  theme(
    panel.background = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    plot.background = element_blank(),
    legend.position = "left"
  ) +
  guides(color = "none")

# Feature Names Only (pp2)
pp2 <- ggplot(importance_data) +
  geom_text(aes(x = 0, y = Feature, label = Feature), 
            hjust = 0.5, color = "black", size = 3) +
  labs(x = NULL, y = NULL) +
  theme_minimal() +
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    panel.grid = element_blank()
  )

# Bar Chart with Gain Values (pp3)
pp3 <- ggplot(importance_data, aes(x = Gain, y = Feature)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(x = "Feature Importance (Gain)", y = NULL) +
  theme_minimal() +
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.text.x = element_text(size = 10),
    plot.title = element_text(size = 14, hjust = 0.5),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.margin = unit(c(0.5, 0.5, 0.5, 0.2), "cm")
  )

# Combine the Plots
combined_plot_psychosis <- plot_grid(
  pp1, pp2, pp3,
  ncol = 3
)

########## Voice-hearers sample

# Partitioning
combined_data <- gen_pop_sample %>%
  filter(ue_ever_unreal_voice == 1) %>%
  mutate(diagnosis = 0) %>%
  bind_rows(
    psychosis_sample %>%
      filter(ue_ever_unreal_voice == 1) %>%
      mutate(diagnosis = 1)
  )

combined_data<- combined_data %>% select(-ue_ever_unreal_voice)

combined_data$diagnosis <- as.factor(combined_data$diagnosis)

set.seed(4435)
combined_data_split_data <- initial_split(
  data = combined_data,
  prop = 0.75,
  strata = "diagnosis"
)

combined_data_training<- training(combined_data_split_data)
combined_data_testing<- testing(combined_data_split_data)

# Voice-hearers: The model

model_recipe <- recipes::recipe(
  diagnosis ~ ., data = combined_data_training) %>%
  update_role(eid, new_role = "ID") %>%
  update_role(age, new_role = "age") %>%
  step_zv(all_predictors()) %>%
  step_nzv(all_predictors()) %>%
  step_range(all_predictors(), -all_factor(),-all_nominal(), min = 0, max = 1) %>%
  step_unknown(all_nominal_predictors(), all_factor_predictors()) %>%  # Handle unknowns for all categorical variables
  step_dummy(all_nominal_predictors(), all_factor_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors()) %>%
  step_nzv(all_predictors())

# 4. cross validaiton  
cv_folds <- vfold_cv(combined_data_training, strata = diagnosis, repeats = 10)

# Create a model and workflow
xgb_model <- boost_tree(
  trees = tune(),
  learn_rate = tune(),
  tree_depth = tune(),
  min_n = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  mtry = tune()
) %>%
  set_mode("classification") %>%
  set_engine("xgboost", scale_pos_weight = tune(), eval_metric = "auc")

workflow.xgboost <- workflow() %>%
  add_recipe(model_recipe) %>%
  add_model(xgb_model)

xgboost_params <- parameters(
  trees(range = c(500, 1500)),
  learn_rate(),
  tree_depth(),
  min_n(),
  loss_reduction(),
  scale_pos_weight(range = c(0.5, 0.6), trans = NULL),
  sample_size = sample_prop(),
  finalize(mtry(), combined_data_training)
)

set.seed(321)
# Start timing
start_time <- Sys.time()
xgboost_tune <- workflow.xgboost %>%
  tune_bayes(
    resamples = cv_folds,
    param_info = xgboost_params,
    iter = 100,
    metrics = metric_set(brier_class, bal_accuracy,mcc,f_meas,recall,precision),
    control = control_bayes(no_improve = 25, save_pred = TRUE, verbose = TRUE),
    initial = 15
  )

end_time <- Sys.time()
end_time - start_time 

# Return results
combined_model_results <- list(
  model_recipe = model_recipe,
  cv_folds = cv_folds,
  xgb_model = xgb_model,
  workflow.xgboost = workflow.xgboost,
  xgboost_params = xgboost_params,
  xgboost_tune = xgboost_tune,
  best_brier_class = select_best(xgboost_tune, metric = "brier_class")
)

# Voice-hearers only sample: Performance

final_xgb <- finalize_workflow(combined_model_results$workflow.xgboost, combined_model_results$best_brier_class)

# Fit the finalized workflow on the training data
fitted_workflow <- final_xgb %>%
  fit(data = combined_data_training)

# Variable Importance Plot
vip(extract_fit_parsnip(fitted_workflow), geom = "point", num_features = 40)

# Recipe Specification
recipe_spec <- fitted_workflow %>%
  extract_recipe()
recipe_spec

# Extract the recipe from the fitted workflow (if needed)
extracted_recipe <- fitted_workflow %>%
  extract_recipe()

print(extracted_recipe)

# Make predictions on the test data
xgb.pred <- predict(fitted_workflow, new_data = combined_data_testing)
print(xgb.pred)

# Assess the Model
# Augment the test data with predictions
set.seed(2)
augment_xgboost <- augment(fitted_workflow, new_data = combined_data_testing)


combined_roc_curve <- augment_xgboost %>%
  roc_curve(truth = diagnosis, .pred_1, event_level = "second") %>%
  autoplot()
print(combined_roc_curve)

# Calculate and store training predictions
set.seed(56)
xgb_training_pred <- 
  predict(fitted_workflow, combined_data_training) %>% 
  bind_cols(predict(fitted_workflow, combined_data_training, type = "prob")) %>% 
  bind_cols(combined_data_training %>% select(diagnosis))

# Calculate and store test predictions
xgb_test_pred <- 
  predict(fitted_workflow, combined_data_testing) %>% 
  bind_cols(predict(fitted_workflow, combined_data_testing, type = "prob")) %>% 
  bind_cols(combined_data_testing %>% select(diagnosis))


# Calculate and print ROC AUC for training and test sets
train_auc <- xgb_training_pred %>%
  roc_auc(truth = diagnosis, .pred_1, event_level = "second")
print(train_auc)

test_auc <- xgb_test_pred %>%
  roc_auc(truth = diagnosis, .pred_1, event_level = "second")
print(test_auc)

# Calculate and print accuracy
train_accuracy <- sum(xgb_training_pred$.pred_class == xgb_training_pred$diagnosis) / nrow(xgb_training_pred)
print(tibble(Metric = "Train accuracy", Value = train_accuracy))

test_accuracy <- sum(xgb_test_pred$.pred_class == xgb_test_pred$diagnosis) / nrow(xgb_test_pred)
print(tibble(Metric = "Test accuracy", Value = test_accuracy))

# Calculate and print Brier scores
train_brier <- brier_class(
  data = xgb_training_pred,
  truth = diagnosis,
  .pred_0
)
print(train_brier)

test_brier <- brier_class(
  data = xgb_test_pred,
  truth = diagnosis,
  .pred_0
)
print(test_brier)

# Calculate and print a_PRC 

a_prc_train <- xgb_training_pred %>%
  pr_auc(truth = diagnosis, 
         .pred_1, 
         event_level = "second")
print(a_prc_train)

a_prc_test <- xgb_test_pred %>%
  pr_auc(truth = diagnosis, 
         .pred_1, 
         event_level = "second")
print(a_prc_test)

# Calculate and print balanced accuracy
train_bal_acc <- xgb_training_pred %>% 
  bal_accuracy(truth = diagnosis, .pred_class , event_level = "second", estimator = "binary")
print(train_bal_acc)

test_bal_acc <- xgb_test_pred %>%
  bal_accuracy(truth = diagnosis, .pred_class , event_level = "second", estimator = "binary")
print(test_bal_acc)

# Calculate and print F1
train_f1<- xgb_training_pred %>% 
  f_meas(truth = diagnosis, .pred_class , event_level = "second", estimator = "binary")
print(train_f1)

test_f1 <- xgb_test_pred %>%
  f_meas(truth = diagnosis, .pred_class , event_level = "second", estimator = "binary")
print(test_f1)

# Calculate and print balanced accuracy
train_mcc<- xgb_training_pred %>% 
  mcc(truth = diagnosis, .pred_class , event_level = "second", estimator = "binary")
print(train_mcc)

test_mcc <- xgb_test_pred %>%
  mcc(truth = diagnosis, .pred_class , event_level = "second", estimator = "binary")
print(test_mcc)

# Calculate and print precision
train_precision<- xgb_training_pred %>% 
  precision(truth = diagnosis, .pred_class , event_level = "second", estimator = "binary")
print(train_precision)

test_precision <- xgb_test_pred %>%
  precision(truth = diagnosis, .pred_class , event_level = "second", estimator = "binary")
print(test_precision)

# Calculate and print precision
train_recall<- xgb_training_pred %>% 
  recall(truth = diagnosis, .pred_class , event_level = "second", estimator = "binary")
print(train_recall)

test_recall <- xgb_test_pred %>%
  recall(truth = diagnosis, .pred_class , event_level = "second", estimator = "binary")
print(test_recall)

test_specificity <- xgb_test_pred %>%
  specificity(truth = diagnosis, .pred_class , event_level = "second", estimator = "binary")
print(test_specificity)

# Create data frames for metrics
train_metrics <- tibble(
  Train_AUC = train_auc,
  Train_Accuracy = train_accuracy,
  Train_Brier_Score = train_brier,
  Train_a_PRC = a_prc_train,
  Train_bal_acc = train_bal_acc,
  Train_F_meas = train_f1,
  Train_mcc = train_mcc,
  Train_precision = train_precision,
  Train_recall = train_recall
)

test_metrics <- tibble(
  Test_AUC = test_auc$.estimate,
  Test_Accuracy = test_accuracy,
  Test_Brier_Score = test_brier$.estimate,
  Test_a_PRC = a_prc_test$.estimate,
  Test_bal_acc = test_bal_acc$.estimate,
  Test_F_meas = test_f1$.estimate,
  Test_mcc = test_mcc$.estimate,
  Test_precision = test_precision$.estimate,
  Test_recall = test_recall$.estimate,
  test_specificity = test_specificity$.estimate
)

combined_roc_curve <- 
  roc_curve(xgb_test_pred, truth = diagnosis, .pred_1, event_level = "second") %>%
  autoplot() +
  theme_minimal() +
  theme(
    panel.grid.major = element_blank(),  
    panel.grid.minor = element_blank(), 
    axis.line = element_line(),         
    axis.ticks = element_line()          
  ) +
  coord_fixed(expand = FALSE)

# Voice-hearers only sample: Performance per iteration
all_train_metrics <- list()  
all_test_metrics <- list()

# Process each fold in the cross-validation for the model
for (fold_index in seq_len(nrow(combined_model_results$cv_folds))) {
  
  # Extract fold and repetition information
  repetition_num <- combined_model_results$cv_folds$id[fold_index]  
  fold_num <- combined_model_results$cv_folds$id2[fold_index]  
  
  split <- combined_model_results$cv_folds$splits[[fold_index]]
  training_fold <- analysis(split)
  testing_fold <- assessment(split)
  
  # Finalize and fit the model
  final_xgb <- finalize_workflow(combined_model_results$workflow.xgboost, combined_model_results$best_brier_class)
  fitted_workflow <- final_xgb %>% fit(data = training_fold)
  
  # Generate predictions for training and test sets
  xgb_training_pred <- predict(fitted_workflow, training_fold) %>%  
    bind_cols(predict(fitted_workflow, training_fold, type = "prob")) %>%  
    bind_cols(training_fold %>% select(diagnosis))
  
  xgb_test_pred <- predict(fitted_workflow, testing_fold) %>% 
    bind_cols(predict(fitted_workflow, testing_fold, type = "prob")) %>% 
    bind_cols(testing_fold %>% select(diagnosis))
  
  # Calculate metrics for training and test sets
  train_auc <- xgb_training_pred %>% roc_auc(truth = diagnosis, .pred_1, event_level = "second")
  test_auc <- xgb_test_pred %>% roc_auc(truth = diagnosis, .pred_1, event_level = "second")
  
  train_accuracy <- xgb_training_pred %>% accuracy(truth = diagnosis, .pred_class, event_level = "second", estimator = "binary")
  test_accuracy <- xgb_test_pred %>% accuracy(truth = diagnosis, .pred_class, event_level = "second", estimator = "binary")
  
  train_brier <- brier_class(data = xgb_training_pred, truth = diagnosis, .pred_0)
  test_brier <- brier_class(data = xgb_test_pred, truth = diagnosis, .pred_0)
  
  a_prc_train <- xgb_training_pred %>% pr_auc(truth = diagnosis, .pred_1, event_level = "second")
  a_prc_test <- xgb_test_pred %>% pr_auc(truth = diagnosis, .pred_1, event_level = "second")
  
  train_bal_acc <- xgb_training_pred %>% bal_accuracy(truth = diagnosis, .pred_class, event_level = "second", estimator = "binary")
  test_bal_acc <- xgb_test_pred %>% bal_accuracy(truth = diagnosis, .pred_class, event_level = "second", estimator = "binary")
  
  train_f1 <- xgb_training_pred %>% f_meas(truth = diagnosis, .pred_class, event_level = "second", estimator = "binary")
  test_f1 <- xgb_test_pred %>% f_meas(truth = diagnosis, .pred_class, event_level = "second", estimator = "binary")
  
  train_mcc <- xgb_training_pred %>% mcc(truth = diagnosis, .pred_class, event_level = "second", estimator = "binary")
  test_mcc <- xgb_test_pred %>% mcc(truth = diagnosis, .pred_class, event_level = "second", estimator = "binary")
  
  train_precision <- xgb_training_pred %>% precision(truth = diagnosis, .pred_class, event_level = "second", estimator = "binary")
  test_precision <- xgb_test_pred %>% precision(truth = diagnosis, .pred_class, event_level = "second", estimator = "binary")
  
  train_recall <- xgb_training_pred %>% recall(truth = diagnosis, .pred_class, event_level = "second", estimator = "binary")
  test_recall <- xgb_test_pred %>% recall(truth = diagnosis, .pred_class, event_level = "second", estimator = "binary")
  
  train_spec <- xgb_training_pred %>% specificity(truth = diagnosis, .pred_class, event_level = "second", estimator = "binary")
  test_spec <- xgb_test_pred %>% specificity(truth = diagnosis, .pred_class, event_level = "second", estimator = "binary")
  
  # Store metrics for each fold
  train_metrics <- tibble(
    Model = 1,  # Directly specify model ID
    Repetition = as.integer(gsub("Repeat", "", repetition_num)),
    Fold = as.integer(gsub("Fold", "", fold_num)),
    Train_AUC = train_auc$.estimate,
    Train_accuracy = train_accuracy$.estimate,
    Train_bal_acc = train_bal_acc$.estimate,
    Train_Brier_Score = train_brier$.estimate,
    Train_a_PRC = a_prc_train$.estimate,
    Train_F_meas = train_f1$.estimate,
    Train_mcc = train_mcc$.estimate,
    Train_precision = train_precision$.estimate,
    Train_recall = train_recall$.estimate,
    Train_spec = train_spec$.estimate
  )
  
  test_metrics <- tibble(
    Model = 1,  # Directly specify model ID
    Repetition = as.integer(gsub("Repeat", "", repetition_num)),
    Fold = as.integer(gsub("Fold", "", fold_num)),
    Test_AUC = test_auc$.estimate,
    Test_accuracy = test_accuracy$.estimate,
    Test_bal_acc = test_bal_acc$.estimate,
    Test_Brier_Score = test_brier$.estimate,
    Test_a_PRC = a_prc_test$.estimate,
    Test_F_meas = test_f1$.estimate,
    Test_mcc = test_mcc$.estimate,
    Test_precision = test_precision$.estimate,
    Test_recall = test_recall$.estimate,
    Test_spec = test_spec$.estimate
  )
  
  
  all_train_metrics[[paste(repetition_num, fold_num, sep = "_")]] <- train_metrics
  all_test_metrics[[paste(repetition_num, fold_num, sep = "_")]] <- test_metrics
  
  message(paste("Fold", fold_num, "in Repetition", repetition_num, "completed for model 1"))
}

# Combine all fold metrics into data frames
train_metrics_table <- bind_rows(all_train_metrics)
test_metrics_table <- bind_rows(all_test_metrics)

# Print final tables for training and testing metrics
print(train_metrics_table)
print(test_metrics_table)

train_summary <- train_metrics_table %>%
  group_by(Model, Repetition) %>%
  summarise(
    Train_AUC_Mean = mean(Train_AUC),
    Train_AUC_SD = sd(Train_AUC),
    Train_accuracy_Mean = mean(Train_accuracy),
    Train_accuracy_SD = sd(Train_accuracy),
    Train_bal_acc_Mean = mean(Train_bal_acc),
    Train_bal_acc_SD = sd(Train_bal_acc),
    Train_Brier_Score_Mean = mean(Train_Brier_Score),
    Train_Brier_Score_SD = sd(Train_Brier_Score),
    Train_a_PRC_Mean = mean(Train_a_PRC),
    Train_a_PRC_SD = sd(Train_a_PRC),
    Train_F_meas_Mean = mean(Train_F_meas),
    Train_F_meas_SD = sd(Train_F_meas),
    Train_mcc_Mean = mean(Train_mcc),
    Train_mcc_SD = sd(Train_mcc),
    Train_precision_Mean = mean(Train_precision),
    Train_precision_SD = sd(Train_precision),
    Train_recall_Mean = mean(Train_recall),
    Train_recall_SD = sd(Train_recall),
    Train_spec_Mean = mean(Train_spec),
    Train_spec_SD = sd(Train_spec),
    .groups = "drop"  # Drop grouping after summarization
  )

test_summary <- test_metrics_table %>%
  group_by(Model, Repetition) %>%
  summarise(
    Test_AUC_Mean = mean(Test_AUC),
    Test_AUC_SD = sd(Test_AUC),
    Test_accuracy_Mean = mean(Test_accuracy),
    Test_accuracy_SD = sd(Test_accuracy),
    Test_bal_acc_Mean = mean(Test_bal_acc),
    Test_bal_acc_SD = sd(Test_bal_acc),
    Test_Brier_Score_Mean = mean(Test_Brier_Score),
    Test_Brier_Score_SD = sd(Test_Brier_Score),
    Test_a_PRC_Mean = mean(Test_a_PRC),
    Test_a_PRC_SD = sd(Test_a_PRC),
    Test_F_meas_Mean = mean(Test_F_meas),
    Test_F_meas_SD = sd(Test_F_meas),
    Test_mcc_Mean = mean(Test_mcc),
    Test_mcc_SD = sd(Test_mcc),
    Test_precision_Mean = mean(Test_precision),
    Test_precision_SD = sd(Test_precision),
    Test_recall_Mean = mean(Test_recall),
    Test_recall_SD = sd(Test_recall),
    Test_spec_Mean = mean(Test_spec),
    Test_spec_SD = sd(Test_spec),
    .groups = "drop"  # Drop grouping after summarization
  )

# Voice-hearers only sample: Threshold adjustment
threshold_data_1 <- xgb_test_pred %>%
  threshold_perf(diagnosis, .pred_0 , thresholds = seq(0.1, 1, by = 0.0025), metric_set(sensitivity, specificity, bal_accuracy, j_index))

threshold_data_1 %>%
  filter(.threshold %in% c(0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 1))

threshold_data_1 <- threshold_data_1 %>%
  filter(.metric != "distance") %>%
  mutate(group = ifelse(.metric %in% c("specificity", "sensitivity", "bal_accuracy","j_index"), "1", "2"))


max_j_index_threshold_1 <- threshold_data_1 %>%
  filter(.metric == "j_index") %>%
  filter(.estimate == max(.estimate)) %>%
  pull(.threshold)

ggplot(threshold_data_1, aes(x = .threshold, y = .estimate, color = .metric, alpha = group)) +
  geom_line() +
  theme_minimal() +
  scale_color_viridis_d(end = 0.9) +
  scale_alpha_manual(values = c(.4, 1), guide = "none") +
  geom_vline(xintercept = max_j_index_threshold_1, alpha = .6, color = "grey30") +
  labs(
    x = "Threshold",
    y = "Metric Estimate",
  )

# Print the best threshold based on the J-Index
threshold_data_1 %>%
  filter(.threshold == max_j_index_threshold_1)

# Print the best threshold based on the J-Index
best_threshold <- threshold_data_1 %>%
  filter(.threshold == max_j_index_threshold_1)

hard_pred_0.785 <- xgb_test_pred %>%
  mutate(
    .pred = make_two_class_pred(
      estimate = .pred_0,
      levels = levels(diagnosis),
      threshold = 0.785
    )
  ) %>%
  select(diagnosis, contains(".pred"))

hard_pred_0.785$.pred <- as.factor(as.character(hard_pred_0.785$.pred))

best_threshold_conf_matrix <- hard_pred_0.785 %>%
  count(.truth = diagnosis, .pred)


hard_pred_0.5 <- xgb_test_pred %>%
  mutate(
    .pred = make_two_class_pred(
      estimate = .pred_0,
      levels = levels(diagnosis),
      threshold = 0.5
    )
  ) %>%
  select(diagnosis, contains(".pred"))

hard_pred_0.5$.pred <- as.factor(as.character(hard_pred_0.5$.pred))

threshold_0.5_conf_matrix <- hard_pred_0.5 %>%
  count(.truth = diagnosis, .pred)

j_index(hard_pred_0.5, diagnosis, .pred)
j_index(hard_pred_0.785, diagnosis, .pred)

paste0("Threshold at 0.785")
print(best_threshold_conf_matrix)

#  paste0("Threshold at 0.5")
print(threshold_0.5_conf_matrix)

# ROC_CURVE

roc_curve(hard_pred_0.785, truth = diagnosis, .pred_1, event_level = "second") %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity)) +
  geom_path() +
  geom_abline(lty = 3) +
  coord_equal() +
  theme_bw()

roc_curve(hard_pred_0.5, truth = diagnosis, .pred_1, event_level = "second") %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity)) +
  geom_path() +
  geom_abline(lty = 3) +
  coord_equal() +
  theme_bw()

# Voice hearers only sample: Performance
hard_pred_0.785 %>%
  roc_curve(truth = diagnosis, .pred_1, event_level = "second")  %>%
  autoplot()


# Calculate and print ROC AUC for training and test sets
test_auc_0.785 <- hard_pred_0.785 %>%
  roc_auc(truth = diagnosis, .pred_1,  event_level = "second")
print(test_auc_0.785)


test_accuracy_0.785 <- hard_pred_0.785 %>%
  accuracy(truth = diagnosis, estimate = .pred)

test_bal_accuracy_0.785 <- hard_pred_0.785 %>%
  bal_accuracy(truth = diagnosis, estimate = .pred)

test_brier_0.785 <- brier_class(
  data = hard_pred_0.785,
  truth = diagnosis,
  .pred_0,
)
print(test_brier_0.785)


a_prc_test_0.785 <- hard_pred_0.785 %>%
  pr_auc(truth = diagnosis, 
         .pred_1, 
         event_level = "second")
print(a_prc_test_0.785)


# Calculate and print F1

test_f1_0.785 <- hard_pred_0.785 %>%
  f_meas(truth = diagnosis, .pred , event_level = "second", estimator = "binary")
print(test_f1_0.785)

# Calculate and print balanced accuracy

test_mcc_0.785 <- hard_pred_0.785 %>%
  mcc(truth = diagnosis, .pred, event_level = "second", estimator = "binary")
print(test_mcc_0.785)

# Calculate and print precision
test_precision_0.785 <- hard_pred_0.785 %>%
  precision(truth = diagnosis, .pred, event_level = "second", estimator = "binary")
print(test_precision_0.785)

# Calculate and print precision

test_recall_0.785 <- hard_pred_0.785 %>%
  recall(truth = diagnosis, .pred , event_level = "second", estimator = "binary")
print(test_recall)

test_spec_0.785 <- hard_pred_0.785 %>%
  specificity(truth = diagnosis, .pred , event_level = "second", estimator = "binary")
print(test_spec_0.785)

test_metrics_0.785 <- tibble(
  Test_AUC = test_auc_0.785$.estimate,
  Test_accuracy = test_accuracy_0.785$.estimate,
  Test_bal_acc = test_bal_accuracy_0.785$.estimate,
  Test_Brier_Score = test_brier_0.785$.estimate,
  Test_a_PRC = a_prc_test_0.785$.estimate,
  Test_F_meas = test_f1_0.785$.estimate,
  Test_mcc = test_mcc_0.785$.estimate,
  Test_precision = test_precision_0.785$.estimate,
  Test_recall = test_recall_0.785$.estimate,
  Test_spec = test_spec_0.785$.estimate
)
print(test_metrics_0.785)

# Voice hearers only: The information gain and the SHAP values

combined_xgboost_shapviz.train <- shapviz::shapviz(
  extract_fit_engine(fitted_workflow), 
  X_pred = bake(
    prep(combined_model_results$model_recipe),
    has_role("predictor"),
    new_data = NULL,
    composition = "matrix"), interactions = TRUE)

combined_xgboost_shapviz.train$S <- -combined_xgboost_shapviz.train$S

# Plot SHAP values

shapviz::sv_importance(combined_xgboost_shapviz.train, kind = "beeswarm", max_display = 10, show_numbers = TRUE)


library(xgboost)

extract_fitted_workflow.train <- extract_fit_engine(fitted_workflow)
importance_scores.train <- xgb.importance(model=extract_fitted_workflow.train)

