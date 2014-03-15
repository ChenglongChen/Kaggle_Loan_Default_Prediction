#########################################
# By Chenglong Chen
# c.chenglong@gmail.com
# http://www.kaggle.com/users/102203/yr
#########################################


rm(list=ls(all=TRUE))
gc(reset=T)

## put the required packages here
require(data.table)
require(bit64)
require(caret)
require(gbm)

## set the working directory to the path that contains all the data files:
# - train_v2.csv
# - test_v2.csv
# - sampleSubmission.csv
# - Defaulter_features.RData
# - LGD_features.RData
setwd('E:/Loan/data')


######################
## Helper Functions ##
######################

#### This function imputes each feature with some constant values
Impute <- function(data, value){

  num <- apply(data, 2, function(x) sum(is.na(x)))

  data <- as.matrix(data)
  data[which(is.na(data))] <- rep(value, num)
  data <- as.data.frame(data)

  return(data)
}


#### This function computes f1-score for varying cutoffs
compF1score <- function(obs, pr, cutoffs){
  
  #### This function computes f1-score for a single cutoff
  compF1scoreSingle <- function(obs, pr, cutoff){
    pred <- ifelse(pr>cutoff, 1, 0)
    true_pos <- sum(obs == 1)
    pred_pos <- sum(pred == 1)
    TP <- sum(obs == 1 & pred == 1)
    precision <- TP/pred_pos
    recall <- TP/true_pos
    f1_score <- 2*precision*recall/(precision+recall)
    return(ifelse(is.na(f1_score), 0, f1_score))
  }
  
  ## We now compute f1-score for varying cutoffs
  f1_scores <- sapply(cutoffs, function(cutoff) compF1scoreSingle(obs, pr, cutoff))
  return(f1_scores)
}


#### This function applies some transformations to the loss
logit_fn <- function(y, y_min, y_max, epsilon, transform='logit'){
  if(transform == 'logit'){
    ## logit transform is suggested in the paper
    # M. Bottai, B. Cai, and R. E. McKeown,
    # "logistic quantile regression for bounded outcomes,"
    # Statistics in Medicine, vol. 29, no. 2, pp. 309-317, 2010.
    # and discussed in
    # http://stats.stackexchange.com/questions/48034/dealing-with-regression-of-unusually-bounded-response-variable
    y1 <- log((y-(y_min-epsilon))/(y_max+epsilon-y))
  }else if(transform == 'logarithm'){
    y1 <- log(y)
  }else if(transform == 'identity'){
    y1 <- y
  }
  return(y1)
}


#### This function applies anti transformations
antilogit_fn <- function(antiy, y_min, y_max, epsilon, transform='logit'){
  if(transform == 'logit'){
    y <- (exp(antiy)*(y_max+epsilon)+y_min-epsilon)/(1+exp(antiy))
  }else if(transform == 'logarithm'){
    y <- exp(antiy)
  }else if(transform == 'identity'){
    y <- antiy
  }
  return(y)
}


###################
## Default Model ##
###################

#### This function trains a gbm as defaulter classifier
trainDefaulterClassifier <- function(loan, gbm_params, plot_on=FALSE){
  
  gc(reset=TRUE)
  
  model <- gbm(
    default ~ .,
    data = loan,
    distribution = 'bernoulli',
    n.trees = gbm_params$n.trees,
    shrinkage = gbm_params$shrinkage,
    interaction.depth = gbm_params$interaction.depth,
    train.fraction = gbm_params$train.fraction,
    bag.fraction = gbm_params$bag.fraction,
    n.minobsinnode = gbm_params$n.minobsinnode,
    cv.folds = gbm_params$cv.folds,
    class.stratify.cv = gbm_params$class.stratify.cv,
    verbose = gbm_params$verbose,
    n.cores = gbm_params$n.cores,
    keep.data = gbm_params$keep.data)
  
  if(plot_on == TRUE){
    par(mfrow=c(1,1))
    gbm.perf(model)
    min.cv.err <- min(model$cv.err, na.rm=T)
    title(main=paste('Minimum cv.err = ', round(min.cv.err, 5), sep=''))
    x <- seq(-gbm_params$n.trees, 2*gbm_params$n.trees)
    y <- rep(min.cv.err, length(x))
    points(x, y, type='l', col='blue', lwd=2, lty=2)
  }
  
  return(model)
  
}


################
## Loss Model ##
################

#### This function trains a gbm for loss given default
trainLossRegressor <- function(loan, gbm_params, plot_on=FALSE){
  
  gc(reset=TRUE)
  
  if(gbm_params$distribution == 'laplace'){
    
    model <- gbm(formula = logit_loss ~ .,
                 data = loan,
                 distribution = 'laplace',
                 n.trees = gbm_params$n.trees,
                 shrinkage = gbm_params$shrinkage,
                 interaction.depth = gbm_params$interaction.depth, 
                 train.fraction = gbm_params$train.fraction,
                 bag.fraction = gbm_params$bag.fraction,
                 n.minobsinnode = gbm_params$n.minobsinnode,
                 cv.folds = gbm_params$cv.folds,
                 verbose = gbm_params$verbose,
                 n.cores = gbm_params$n.cores,
                 keep.data = gbm_params$keep.data)
    
    if(plot_on == TRUE){
      par(mfrow=c(1,1))
      gbm.perf(model)
      min.cv.err <- min(model$cv.err, na.rm=T)
      title(main=paste('Minimum cv.err = ', round(min.cv.err, 5), sep=''))
      x <- seq(-gbm_params$n.trees, 2*gbm_params$n.trees)
      y <- rep(min.cv.err, length(x))
      points(x, y, type='l', col='blue', lwd=2, lty=2)
    }
    
  }else if(gbm_params$distribution == 'quantile'){
    
    model <- gbm(formula = logit_loss ~ .,
                 data = loan,
                 distribution = list(name="quantile", alpha=0.5),
                 n.trees = gbm_params$n.trees,
                 shrinkage = gbm_params$shrinkage, 
                 interaction.depth = gbm_params$interaction.depth, 
                 train.fraction = gbm_params$train.fraction,
                 bag.fraction = gbm_params$bag.fraction,
                 n.minobsinnode = gbm_params$n.minobsinnode,
                 cv.folds = gbm_params$cv.folds,
                 verbose = gbm_params$verbose,
                 n.cores = gbm_params$n.cores,
                 keep.data = gbm_params$keep.data)
    
    if(plot_on == TRUE){
      par(mfrow=c(1,1))
      gbm.perf(model)
      min.cv.err <- min(model$cv.err, na.rm=T)
      title(main=paste('Minimum cv.err = ', round(min.cv.err, 5), sep=''))
      x <- seq(-gbm_params$n.trees, 2*gbm_params$n.trees)
      y <- rep(min.cv.err, length(x))
      points(x, y, type='l', col='blue', lwd=2, lty=2)
    }
    
  }
  
  return(model)  
}


########################
## Training & Testing ##
########################

#### This function uses cv to choose the best cutoff
cvForParams <- function(loan, gbm_params_default, default_predictors,
                        gbm_params_loss, loss_predictors, transform, cutoffs,
                        stratified, repeated_cv=5, cv_folds=5, random_seed=2014){
  gc(reset=TRUE)
  
  ## the params for the transformation, they are only used when transfom='logit'
  y_min <- 1
  y_max <- 100
  epsilon <- 0.001
  
  f1_score <- rep(0, length(cutoffs))
  
  ## random seed to ensure reproduciable results
  set.seed(random_seed)
  for(iter in seq(1, repeated_cv)){
    
    
    #### split training/testing set
    ## we use 80/20 splits for training/testing set
    ratio1 <- 0.80 ## ratio to split the wholde data set into training/testing set
    ratio2 <- 60/80 ## ratio to split training into training2/valid set
    if(stratified == TRUE){
      ## stratified split
      split1 <- createDataPartition(loan$default, p = ratio1)[[1]]
    }else{
      ## unstratified split
      split1 <- sample(length(loan$default), floor(ratio1*length(loan$default)))
    }
    ## this now contains 80% data for training, and 20% data for testing
    training <- loan[split1,]
    testing <- loan[-split1,]
    
    
    for(fold in seq(1, cv_folds)){
      
      cat('Iteration: ', iter,
          ' | Fold: ', fold,
          '\n', sep='')
      
      #### split training set into training2/valid set
      ## we use 60/20 splits for training2/valid set
      if(stratified == TRUE){
        ## stratified split
        split2 <- createDataPartition(training$default, p = ratio2)[[1]]
      }else{
        ## unstratified split
        split2 <- sample(length(training$default), floor(ratio2*length(training$default)))
      }      
      ## this now contains 60% data for training2, and 20% data for valid
      training2 <- training[split2,]
      valid <- training[-split2,]      
      

      #### We first train defaulter classifier
      cat('Train defaulter classifier.\n')
      model_default <- trainDefaulterClassifier(
        loan = training2[,c(default_predictors, 'default')],
        gbm_params = gbm_params_default)
      
      
      cat('Done.\n')
      
      
      ####
      cat('Compute f1-score on valid set for varying cutoffs.\n')
      ## make prediction on the valid set
      valid$loss_pr <- predict(model_default, newdata=valid)
      ## compute the f1-score for varying cutoffs
      this_f1_score <- compF1score(valid$default, valid$loss_pr, cutoffs)
      ## accumulate the f1-score for this iteration-fold
      f1_score <- f1_score + this_f1_score

      
      cat('Done.\n')
      cat('Iteration: ', iter,
          ' | Fold: ', fold,
          ' Done. \n', sep='')
      
    }
  }
  
  # aveage over repeated_cv & cv_folds
  f1_score <- f1_score/(repeated_cv*cv_folds)
  # find the best f1-score
  best_f1_score <- max(f1_score)
  # and the corresponding cutoff
  best_cutoff <- cutoffs[which.max(f1_score)]
  # the params for logit-transform
  best_logit_params <- list(y_min, y_max, epsilon)
  
  cat('Best f1-score on valid set: ', best_f1_score, '\n',
      'Best cutoff: ', best_cutoff, '\n', sep='')
  
  ##
  return(list(f1_score, best_cutoff, best_logit_params))
}



#### This function evaluates the performance on the held out testing set
cvForPerformance <- function(loan, gbm_params_default, default_predictors, 
                             gbm_params_loss, loss_predictors, transform, best_cutoff,
                             stratified, repeated_cv, random_seed=2014, logit_params){
  
  gc(reset=TRUE)
  
  # MAE estimated on the hold out testing set during each k-fold cv
  MAE_testing <- rep(0, repeated_cv)
  
  # random seed to ensure reproduciable results
  # we use the same random seed as that in the cv for best cutoff
  # in this case, the testing set is not used in the cv for best cutoff
  # and it is a hold out set that can be used to evaluate the performance
  set.seed(random_seed)
  
  for(iter in seq(1, repeated_cv)){
    
    cat('Iteration: ', iter, '\n')
    
    #### split training/testing set
    ## we use 80/20 splits for training/testing set
    ratio1 <- 0.80 ## ratio to split the wholde data set into training/testing set
    ratio2 <- 60/80 ## ratio to split training into training2/valid set
    if(stratified == TRUE){
      ## stratified split
      split1 <- createDataPartition(loan$default, p = ratio1)[[1]]
    }else{
      ## unstratified split
      split1 <- sample(length(loan$default), floor(ratio1*length(loan$default)))
    }
    ## this now contains 80% data for training, and 20% data for testing
    training <- loan[split1,]
    testing <- loan[-split1,]
    
    
    #### We first train defaulter classifier
    cat('Train defaulter classifier.\n')
    model_default <- trainDefaulterClassifier(
      loan = training[,c(default_predictors, 'default')],
      gbm_params = gbm_params_default)
    
    ## compute the probability of default as additional feature
    training$loss_pr <- predict(model_default, newdata=training)
    
    cat('Done.\n')
    
    
    #### We then train loss given default regressor
    cat('Train LGD regressor.\n')
    ## only use those samples with loss>0
    ind <- which(training$loss>0)
    ## apply transform to the loss
    y_min <- logit_params[[1]]
    y_max <- logit_params[[2]]
    epsilon <- logit_params[[3]]
    training$logit_loss <- training$loss
    ## only for those loss>0
    training$logit_loss[ind] <- logit_fn(training$loss[ind],
                                         y_min, y_max, epsilon, transform)
    
    ## train the loss model
    model_loss <- trainLossRegressor(
      loan = training[ind, c(loss_predictors, 'logit_loss')],
      gbm_params = gbm_params_loss)
    
    cat('Done.\n')
    
    
    #### compute MAE on the testing set
    cat('Compute MAE on testing set.\n')
    ## make prediction on the valid set
    testing$loss_pr <- predict(model_default, newdata=testing)
    
    default <- ifelse(testing$loss_pr>best_cutoff, 1, 0)
    loss_pred_testing <- rep(0, length(testing$loss))
    ind <- which(default==1)
    loss_pred_testing[ind] <- predict(model_loss, newdata=testing[ind,])
    
    
    ## anti transform
    loss_pred_testing[ind] <- antilogit_fn(loss_pred_testing[ind],
                                           y_min, y_max, epsilon, transform)
    loss_pred_testing[loss_pred_testing<0] <- 0
    loss_pred_testing[loss_pred_testing>100] <- 100
    
    MAE_testing[iter] <- mean(abs(testing$loss - loss_pred_testing))
    
    cat('Done.\n')
  }
  
  ## visualization of the distribution of the observed loss vs. the predicted loss
  par(mfrow=c(2,1))
  observed_loss <- testing$loss[testing$loss>0]
  predicted_loss<- loss_pred_testing[testing$loss>0]
  hist(observed_loss, breaks=seq(0,100), freq=FALSE,
       main=paste('Median: ', median(observed_loss), sep=''))
  hist(predicted_loss, breaks=seq(0,100), freq=FALSE,
       main=paste('Median: ', round(median(predicted_loss), 2), sep=''))
  
  ##
  return(MAE_testing)
}



#### This function makes the final submission
makeSubmission <- function(loan, Xtest, gbm_params_default, default_predictors, 
                           gbm_params_loss, loss_predictors, transform, best_cutoff,
                           random_seed=2014, saveFileName, logit_params){
  
  gc(reset=TRUE)
  # random seed to ensure reproduciable results
  set.seed(random_seed)

  #### We first train defaulter classifier
  cat('Train defaulter classifier.\n')
  model_default <- trainDefaulterClassifier(
    loan = loan[,c(default_predictors, 'default')],
    gbm_params = gbm_params_default)
  
  ## compute the probability of default as additional feature
  loan$loss_pr <- predict(model_default, newdata=loan)
  
  cat('Done.\n')
  
  
  #### We then train loss given default regressor
  cat('Train LGD regressor.\n')
  ## only use those samples with loss>0
  ind <- which(loan$loss>0)
  ## loss transform
  y_min <- logit_params[[1]]
  y_max <- logit_params[[2]]
  epsilon <- logit_params[[3]]
  loan$logit_loss <- loan$loss
  ## only for those loss>0
  loan$logit_loss[ind] <- logit_fn(loan$loss[ind],
                                   y_min, y_max, epsilon, transform)
  
  ## train the loss model
  model_loss <- trainLossRegressor(
    loan = loan[ind, c(loss_predictors, 'logit_loss')],
    gbm_params = gbm_params_loss)
  
  cat('Done.\n')

  
  ## make prediction on the testing set
  Xtest$loss_pr <- predict(model_default, newdata=Xtest)
  ## use the best threhold to determin which is probably a defaulter
  loss_pred_sub <- ifelse(Xtest$loss_pr>best_cutoff, 1, 0)
  ## for those defaulter, we predict their loss
  ind <- which(loss_pred_sub == 1)
  
  ## to avoid memory issue we use mini-batch
  len <- length(ind)
  num_batch <- 10
  batch_size <-  floor(len/num_batch)
  
  for(i in seq(1,num_batch)){
    j <- ((i-1)*batch_size+1):(i*batch_size)
    loss_pred_sub[ind[j]] <- predict(model_loss, newdata=Xtest[ind[j],])
  }
  if((num_batch*batch_size+1)<=len){
    j <- (num_batch*batch_size+1):len
    loss_pred_sub[ind[j]] <- predict(model_loss, newdata=Xtest[ind[j],])
  }
  
  ## anti transform
  loss_pred_sub[ind] <- antilogit_fn(loss_pred_sub[ind],
                                     y_min, y_max, epsilon, transform)
  loss_pred_sub[loss_pred_sub<0] <- 0
  loss_pred_sub[loss_pred_sub>100] <- 100
  
  ## make submission
  sub <- read.csv('sampleSubmission.csv', header=T)
  sub$loss <- loss_pred_sub
  write.csv(sub, saveFileName, row.names=F, quote=FALSE)
  
}



##########
## Main ##
##########

#### load training and testing data
# fread is much faster than read.csv
loan <- as.data.frame(fread("./train_v2.csv", header=TRUE, sep=","))
Xtest <- as.data.frame(fread("./test_v2.csv", header=TRUE, sep=","))

#### missing data imputation
med <- apply(loan, 2, median, na.rm=TRUE)
loan <- Impute(loan, med)
Xtest <- Impute(Xtest, med[1:dim(Xtest)[2]])


#### some features
# the default status
loan$default <- ifelse(loan$loss>0, 1, 0)

loan$f274_f527 <- loan$f274 - loan$f527
loan$f274_f528 <- loan$f274 - loan$f528
loan$f527_f528 <- loan$f527 - loan$f528

Xtest$f274_f527 <- Xtest$f274 - Xtest$f527
Xtest$f274_f528 <- Xtest$f274 - Xtest$f528
Xtest$f527_f528 <- Xtest$f527 - Xtest$f528


#### features for default model 
load('Default_features.RData')
# use only the first 15 features that I found
default_predictors <- default_predictors[1:15]


#### features for the loss model
load('LGD_features.RData')


#### to save memory, we delete those unused features
all_predictors <- unique(c(default_predictors, loss_predictors))
all_predictors <- all_predictors[-which(all_predictors == 'loss_pr')]
target <- c('default', 'loss')
loan <- loan[, c(all_predictors, target)]
Xtest <- Xtest[, all_predictors]
gc(reset=TRUE)


#### setup for the cv
# k-fold cv
cv_folds <- 5
# times of performing k-fold cv
repeated_cv <- 10
# number of random seed
seed_num <- 20
# random seeds
set.seed(2014) # to ensure reproducable results
random_seeds <- sample(10000, seed_num)


# the estimated cv MAE for each seed and cv
MAE_testing <- matrix(0, seed_num, repeated_cv)


# varying cutoffs
step <- 0.001
cutoffs <- seq(0.0, 1.0, step)


# stratified sampling or uniformly sampling?
stratified <- TRUE


## what kind of transform you want to apply to the loss
## logit/logarithm helps to boost the performance
#transform <- 'identity'
#transform <- 'logit'
transform <- 'logarithm'


## gbm params for default model
# you can play around with this params
gbm_params_default <- data.frame(distribution = 'bernoulli',
                                 n.trees = 3000,
                                 shrinkage = 0.02,
                                 interaction.depth = 10,
                                 train.fraction = 1.0,
                                 bag.fraction = 0.5,
                                 n.minobsinnode = 10,
                                 cv.folds = 2,
                                 class.stratify.cv = TRUE,
                                 verbose = TRUE,
                                 n.cores = 2,
                                 keep.data = FALSE)

## gbm params for loss model
# For logit transform, I use n.trees = 3000, for logarithm transform
# I use n.trees = 4000. It seems with logarithm transformation, we need
# more than 3000 or even 5000 iterations to arrive at the minimun cv.err.
# To see this, you can turn on the plot_on in cvForParams/cvForPerformance.
# That will plot the training error and cv.err for each iteration. Then you
# can decide if more iterations is perferable.
# I also tried distribution = 'quantile', but I didn't see a big difference.
gbm_params_loss <- data.frame(distribution = 'laplace',
                              n.trees = 4000,
                              shrinkage = 0.02,
                              interaction.depth = 10,
                              train.fraction = 1.0,
                              bag.fraction = 0.5,
                              n.minobsinnode = 10,
                              cv.folds = 2,
                              verbose = TRUE,
                              n.cores = 2,
                              keep.data = FALSE)


########################################################################
# The following are some of the results I obtained for the above params
# setting. (If you change the default_predictors or gbm_params_default
# you'd better use cv to find the corresponding best cutoff.)
# If you want to save your time, just use the best cutoffs as listed
# below for different random seeds.
#
# NOTE: Use the first random seed, run with both logit and logarithm 
# transformations. After taking the average of those two prediction,
# you should yield score around 0.451xx.
########################################################################
# For 15 defaulter features, seed 1 [i.e., 2859]
# Best f1-score on valid set: 0.9488283
# Best cutoff:0.013
########################################################################
# For 15 defaulter features, seed 2 [i.e., 1689]
# Best f1-score on valid set: 0.94864
# Best cutoff:0.005
########################################################################
# For 15 defaulter features, seed 3 [i.e., 6258]
# Best f1-score on valid set: 0.9501886 
# Best cutoff: 0.003 
########################################################################
# For 15 defaulter features, seed 4 [i.e., 3096]
# Best f1-score on valid set: 0.9493345 
# Best cutoff: 0.005 
########################################################################

## some best cutoffs I obtained
best_cutoffs <- c(0.013, 0.005, 0.003, 0.005)
best_logit_params <- list(1, 100, 0.001)


########################################################################
# Since we are using the stochastic version of GBM, run the following for
# varying random seeds and then take the average predictions of them
# to reduce the variance (slightly). As a side product of using different
# random seeds, we now have seed_num*repeated_cv estimates of MAE, which
# can result in a better evaluation of our model performance.

count_seed <- 0
for(seed in random_seeds){
  
  gc(reset=T)
  count_seed <- count_seed + 1
  best_cutoff <- best_cutoffs[count_seed]
  
  # It is quite time consuming to run the whole procedure for many random seeds
  # on my laptop. So, I only run for the first 4 seeds. However, I observe that
  # 4 is enough for an accurate estimate of the CV MAE.
  if(count_seed <= 1){
    cat('The ', count_seed, ' seed...', '\n', sep='')
    
    
    #### cross-validate to find the best cutoff
    # If you want to use the above found best cutoffs, you can comment out
    # this part to save time.
    results_valid <- cvForParams(
      loan, gbm_params_default, default_predictors,
      gbm_params_loss, loss_predictors, transform, cutoffs,
      stratified, repeated_cv, cv_folds, seed
      )
    
    f1_score_valid <- results_valid[[1]]    
    best_cutoff <- results_valid[[2]]
    best_logit_params <- results_valid[[3]]
    
    
    #### cross-validate to estimate MAE using the found best cutoff
    # Since the comp is ended and there is no longer a limit of submission, 
    # I guess you don't need to run this part to see the performance of the model.
    # However, you will need it if you want to have a sense to what extend the
    # local CV different from the public/private LB. BTW, it's around 0.025~0.03.
    MAE_testing[count_seed,] <- cvForPerformance(
      loan, gbm_params_default, default_predictors, 
      gbm_params_loss, loss_predictors, transform, best_cutoff,
      stratified, repeated_cv, seed, best_logit_params
      )
    
    
    #save(list=c('MAE_testing'), file='../Submission/MAE_testing.RData')
    
    #### retrain the model and make a submission
    # If you just want to make a submission, use the best_cutoffs I list above
    # and run the following code only.
    saveFileName <- paste('./submission_',
                          '[Stratified_', stratified, ']_',
                          '[GBM3000_clf_lr0.02]_',
                          '[GBM4000_reg_lr0.02]_',
                          '[BestCutoff_', best_cutoff, ']_',
                          '[Seed', seed, ']_',
                          '[Mean', round(mean(MAE_testing[count_seed,]),5),
                          '_SD', round(sd(MAE_testing[count_seed,]),5),']',
                          '.csv', sep='')
    
    makeSubmission(loan, Xtest, gbm_params_default, default_predictors, 
                   gbm_params_loss, loss_predictors, transform, best_cutoff,
                   seed, saveFileName, best_logit_params)

  } 
}
