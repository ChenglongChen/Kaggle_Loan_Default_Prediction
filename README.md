# Kaggle's Loan Default Prediction - Imperial College London

This is the R code I used to make my submission to Kaggle's Loan Default Prediction - Imperial College London competition. 

My best entry yields 0.45135 on the private LB (0.45185 on the public one), ranking 9 out of 677 participating teams. Using the code here, you can yield similar score. Below, I present a brief description and instruction. The code itself contains lots of comments. So, you'd better see there for details.

It's a two-step approach with a default classifier followed by a loss given default (LGD) regressor. Gradient Boosting Machine (gbm package in R) is 
used to build both the classification and regression part. 
* For the defaulter classifier, I use the golden features revealed by Yasser (thanks a ton!), i.e., f527 and f528, and some other features I found using a GBM. They are stored in the file Defaulter_features.RData in decreasing order with respect to the feature importance returned by a GBM. In specific, I train a GBM with all the raw features and the golden features (use the difference instead, i.e., f274-f528, f274-f527, and f527-f528), then sort it with respect to the feature importance. I have played around with the first 10, 15, and 20 features for classifying defaulter, and found that 15 features probably work best, with a f1-score aroud 0.9488x (sometimes can go to 0.95).
* For the LGD regressor, 107 features are selected using a GBM (similar as we did in selecting features for default model). They are stored in the file LGD_features.RData. In loss regression, I have found that applying logit/logarithm transformation to the loss can help to boost the performance. My best entry is an average of two predictions that use logit and logarithm transformations.


## Requirements

You should have packages: data.table, bit64, gbm, and caret installed in R.


## Instructions

* Download data from competition website: https://www.kaggle.com/c/loan-default-prediction/
* Download both Defaulter_features.RData and LGD_features.RData
* Set the working directory to the path that contains those data files
* Run loan_default_prediction.R
