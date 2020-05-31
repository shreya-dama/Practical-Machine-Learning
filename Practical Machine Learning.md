---
title: "Practical Machine Learning"
---
DATASETS USED:

       1) Training dataset: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
       2) Testing dataset: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv
       
IMPORTING LIBRARIES:


```r
rm(list=ls())                
library(knitr)
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(rpart)
library(rpart.plot)
library(rattle)
```

```
## Rattle: A free graphical interface for data science with R.
## Version 5.3.0 Copyright (c) 2006-2018 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
library(randomForest)
```

```
## randomForest 4.6-14
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:rattle':
## 
##     importance
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
library(corrplot)
```

```
## corrplot 0.84 loaded
```

```r
set.seed(123456)
```

DATA LOADING AND CLEANING:

The next step is loading the dataset from the URL provided above. The training dataset is then partinioned in 2 to create a Training set (70% of the data) for the modeling process and a Test set (with the remaining 30%) for the validations. The testing dataset is not changed and will only be used for the quiz results generation.


```r
UrlTrain <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
UrlTest  <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"


training <- read.csv(url(UrlTrain))
testing  <- read.csv(url(UrlTest))


inTrain  <- createDataPartition(training$classe, p=0.7, list=FALSE)
TrainSet <- training[inTrain, ]
TestSet  <- training[-inTrain, ]
dim(TrainSet)
```

```
## [1] 13737   160
```


```r
dim(TestSet)
```

```
## [1] 5885  160
```
Both created datasets have 160 variables. Those variables have plenty of NA, that can be removed with the cleaning procedures below. The Near Zero variance (NZV) variables are also removed and the ID variables as well

```r
NZV <- nearZeroVar(TrainSet)
TrainSet <- TrainSet[, -NZV]
TestSet  <- TestSet[, -NZV]
dim(TrainSet)
```

```
## [1] 13737   108
```


```r
dim(TestSet)
```

```
## [1] 5885  108
```


```r
AllNA    <- sapply(TrainSet, function(x) mean(is.na(x))) > 0.95
TrainSet <- TrainSet[, AllNA==FALSE]
TestSet  <- TestSet[, AllNA==FALSE]
dim(TrainSet)
```

```
## [1] 13737    59
```


```r
dim(TestSet)
```

```
## [1] 5885   59
```


```r
TrainSet <- TrainSet[, -(1:5)]
TestSet  <- TestSet[, -(1:5)]
dim(TrainSet)
```

```
## [1] 13737    54
```


```r
dim(TestSet)
```

```
## [1] 5885   54
```

THE CORRELATION ANALYSIS:

A correlation among variables is analysed before proceeding to the modeling procedures.

```r
corMatrix <- cor(TrainSet[, -54])
corrplot(corMatrix, order = "FPC", method = "color", type = "lower", 
         tl.cex = 0.8, tl.col = rgb(0, 0, 0))
```

![plot of chunk unnamed-chunk-10](figure/unnamed-chunk-10-1.png)
The highly correlated variables are shown in dark colors in the graph above. To make an evem more compact analysis, a PCA (Principal Components Analysis) could be performed as pre-processing step to the datasets. Nevertheless, as the correlations are quite few, this step will not be applied for this assignment.

PREDICTION MODEL ANALYSIS:

Three methods will be applied to model the regressions (in the Train dataset) and the best one (with higher accuracy when applied to the Test dataset) will be used for the quiz predictions. The methods are: Random Forests, Decision Tree and Generalized Boosted Model, as described below. A Confusion Matrix is plotted at the end of each analysis to better visualize the accuracy of the models.

1) RANDOM FOREST METHOD

```r
set.seed(12345)
controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE)
modFitRandForest <- train(classe ~ ., data=TrainSet, method="rf",
                          trControl=controlRF)
modFitRandForest$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.22%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3904    1    0    0    1 0.0005120328
## B    7 2646    3    2    0 0.0045146727
## C    0    4 2392    0    0 0.0016694491
## D    0    0    6 2245    1 0.0031083481
## E    0    0    0    5 2520 0.0019801980
```


```r
predictRandForest <- predict(modFitRandForest, newdata=TestSet)
confMatRandForest <- confusionMatrix(predictRandForest, TestSet$classe)
confMatRandForest
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    3    0    0    0
##          B    0 1134    0    0    0
##          C    0    2 1026    7    0
##          D    0    0    0  956    4
##          E    0    0    0    1 1078
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9971          
##                  95% CI : (0.9954, 0.9983)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9963          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9956   1.0000   0.9917   0.9963
## Specificity            0.9993   1.0000   0.9981   0.9992   0.9998
## Pos Pred Value         0.9982   1.0000   0.9913   0.9958   0.9991
## Neg Pred Value         1.0000   0.9989   1.0000   0.9984   0.9992
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1927   0.1743   0.1624   0.1832
## Detection Prevalence   0.2850   0.1927   0.1759   0.1631   0.1833
## Balanced Accuracy      0.9996   0.9978   0.9991   0.9954   0.9980
```


```r
plot(confMatRandForest$table, col = confMatRandForest$byClass, 
     main = paste("Random Forest - Accuracy =",
     round(confMatRandForest$overall['Accuracy'], 4)))
```

![plot of chunk unnamed-chunk-13](figure/unnamed-chunk-13-1.png)

2) DECISION TREE METHOD

```r
set.seed(123456)
modFitDecTree <- rpart(classe ~ ., data=TrainSet, method="class")
fancyRpartPlot(modFitDecTree)
```

```
## Warning: labs do not fit even at cex 0.15, there may be some overplotting
```

![plot of chunk unnamed-chunk-14](figure/unnamed-chunk-14-1.png)


```r
predictDecTree <- predict(modFitDecTree, newdata=TestSet, type="class")
confMatDecTree <- confusionMatrix(predictDecTree, TestSet$classe)
confMatDecTree
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1476  206   45   37   30
##          B   50  703   69   79   50
##          C    6   54  821  142   59
##          D   92  134   60  612  135
##          E   50   42   31   94  808
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7511          
##                  95% CI : (0.7398, 0.7621)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6846          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8817   0.6172   0.8002   0.6349   0.7468
## Specificity            0.9245   0.9477   0.9463   0.9144   0.9548
## Pos Pred Value         0.8227   0.7392   0.7588   0.5924   0.7883
## Neg Pred Value         0.9516   0.9116   0.9573   0.9275   0.9436
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2508   0.1195   0.1395   0.1040   0.1373
## Detection Prevalence   0.3048   0.1616   0.1839   0.1755   0.1742
## Balanced Accuracy      0.9031   0.7825   0.8732   0.7747   0.8508
```


```r
plot(confMatDecTree$table, col = confMatDecTree$byClass, 
     main = paste("Decision Tree - Accuracy =",
                  round(confMatDecTree$overall['Accuracy'], 4)))
```

![plot of chunk unnamed-chunk-16](figure/unnamed-chunk-16-1.png)

3) GENERALIZED BOOSTED MODEL(GBM) METHOD 

```r
set.seed(123456)
controlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modFitGBM  <- train(classe ~ ., data=TrainSet, method = "gbm",
                    trControl = controlGBM, verbose = FALSE)
modFitGBM$finalModel
```

```
## A gradient boosted model with multinomial loss function.
## 150 iterations were performed.
## There were 53 predictors of which 53 had non-zero influence.
```


```r
predictGBM <- predict(modFitGBM, newdata=TestSet)
confMatGBM <- confusionMatrix(predictGBM, TestSet$classe)
confMatGBM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1669   13    0    1    0
##          B    4 1116    6    5    5
##          C    0   10 1013   11    3
##          D    1    0    7  942   15
##          E    0    0    0    5 1059
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9854         
##                  95% CI : (0.982, 0.9883)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9815         
##                                          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9970   0.9798   0.9873   0.9772   0.9787
## Specificity            0.9967   0.9958   0.9951   0.9953   0.9990
## Pos Pred Value         0.9917   0.9824   0.9769   0.9762   0.9953
## Neg Pred Value         0.9988   0.9952   0.9973   0.9955   0.9952
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2836   0.1896   0.1721   0.1601   0.1799
## Detection Prevalence   0.2860   0.1930   0.1762   0.1640   0.1808
## Balanced Accuracy      0.9968   0.9878   0.9912   0.9863   0.9889
```


```r
plot(confMatGBM$table, col = confMatGBM$byClass, 
     main = paste("GBM - Accuracy =", round(confMatGBM$overall['Accuracy'], 4)))
```

![plot of chunk unnamed-chunk-19](figure/unnamed-chunk-19-1.png)

APPLYING THE BEST MODEL TO THE DATASET:

The accuracy of the 3 regression modeling methods above are:

Random Forest : 0.9971 Decision Tree : 0.7511 GBM : 0.9854 In that case, the Random Forest model will be applied to predict the 20 quiz results (testing dataset) as shown below


```r
predictTEST <- predict(modFitRandForest, newdata=testing)
predictTEST
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

