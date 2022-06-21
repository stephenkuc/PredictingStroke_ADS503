
library(caret)
library(e1071)
library(Hmisc)
library(corrplot)
library(plyr)
library(pROC)
library(earth)
library(kernlab)

stroke<- read.csv('c:\\maha\\503\\healthcare-dataset-stroke-data.csv', header = TRUE)

str(stroke)

stroke$gender <- as.numeric(as.factor(stroke$gender))
stroke$ever_married <- as.numeric(as.factor(stroke$ever_married))
stroke$work_type <- as.numeric(as.factor(stroke$work_type))
stroke$Residence_type <- as.numeric(as.factor(stroke$Residence_type))
stroke$smoking_status <- as.numeric(as.factor(stroke$smoking_status))
stroke$bmi <- as.numeric(stroke$bmi)
levels(stroke$stroke) <- c("YES", "NO")
#stroke$stroke <- as.factor(stroke$stroke)

stroke$stroke<-ifelse(stroke$stroke == 1,"YES","NO")
table(stroke$stroke)

trainingRows <- createDataPartition(stroke$stroke, p = .80, list = FALSE)
stroke_train <- stroke[trainingRows, ]
stroke_test <- stroke[-trainingRows, ]
stroke_trainx <- stroke_train[,1:11]
stroke_testx <- stroke_test[, 1:11]
stroke_testy <- stroke_test[, 12]
stroke_trainY <- stroke_train[,12]
stroke_trainy <- as.factor(stroke_train[, 12])
stroke_trainimp <- preProcess(stroke_trainx, "knnImpute")
stroke_trainxpr <- predict(stroke_trainimp, stroke_trainx)
stroke_testxpr <- predict(stroke_trainimp, stroke_test)

ctrl <- trainControl(method = "cv",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)


set.seed(500)
ldaFit_stroke <- train(x = stroke_trainxpr,
                       y = stroke_trainy,
                       method = "lda",
                       preProc = c("center","scale"),
                       metric = "ROC",
                       trControl = ctrl)
ldaFit_stroke

glmnGrid <- expand.grid(alpha = c(0, .1, .2, .4, .6, .8, 1),
                        lambda = seq(.01, .2, length = 10))
set.seed(500)
glmnFit_stroke <- train(x = stroke_trainxpr,
                        y = stroke_trainy,
                        method = "glmnet",
                        tuneGrid = glmnGrid,
                        preProc = c("center", "scale"),
                        metric = "ROC",
                        trControl = ctrl)
glmnFit_stroke

set.seed(500)
nscFit_stroke <- train(x = stroke_trainxpr,
                       y = stroke_trainy,
                       method = "pam",
                       preProc = c("center", "scale"),
                       tuneGrid = data.frame(threshold = seq(0, 25, length = 30)),
                       metric = "ROC",
                       trControl = ctrl)
nscFit_stroke

set.seed(500)
indx <- createFolds(stroke_trainy, returnTrain = TRUE)
ctrl <- trainControl(method = "cv", index = indx)
nnetGrid <- expand.grid(decay = c(0, 0.01, .1), 
                        size = c(3, 7, 11, 13))

table(stroke_trainy)

sapply(stroke_trainxpr, function(x) sum(is.na(x)))
nearZeroVar(stroke_trainxpr)
findCorrelation(cor(stroke_trainxpr), cutoff = .75)

set.seed(500)
options(warn = -1)
stroke_nnetTune <- train(x = stroke_trainxpr, y = stroke_trainY,
                         method = "nnet",
                         tuneGrid = nnetGrid,
                         trControl = ctrl,
                         linout = TRUE,
                         trace = FALSE,
                         MaxNWts = 13 * (ncol(stroke_trainxpr) + 1) + 13 + 1,
                         maxit = 1000)
stroke_nnetTune

set.seed(500)
stroke_marsTune <- train(x = stroke_trainxpr, y = stroke_trainy,
                         method = "earth",
                         tuneGrid = expand.grid(degree = 1, nprune = 2:38),
                         trControl = ctrl)
stroke_marsTune
stroke_marsTune$finalModel

set.seed(500)
stroke_svmRTune <- train(x = stroke_trainxpr, y = stroke_trainy,
                         method = "svmRadial",
                         preProc = c("center", "scale"),
                         tuneLength = 14,
                         trControl = ctrl)
stroke_svmRTune

svmGrid <- expand.grid(degree = 1:2, scale = c(0.01, 0.005, 0.001), C = 2^(-2:5))

set.seed(500)
#options(warn=-1)
stroke_svmPTune <- train(x = stroke_trainxpr, y = stroke_trainy,
                         method = "svmPoly",
                         preProc = c("center", "scale"),
                         scale = TRUE,
                         tuneGrid = svmGrid,
                         trControl = ctrl)
stroke_svmPTune

set.seed(500)
stroke_knnTune <- train(x = stroke_trainxpr, y = stroke_trainy,
                        method = "knn",
                        preProc = c("center", "scale"),
                        tuneGrid = data.frame(k = 1:100),
                        trControl = ctrl)
stroke_knnTune

testResults_stroke <- data.frame(OBS = stroke_testy, LDA = predict(ldaFit_stroke, stroke_testxpr))
testResults_stroke$GLMN <- predict(glmnFit_stroke, stroke_testxpr)
testResults_stroke$NSC <- predict(nscFit_stroke, stroke_testxpr)
testResults_stroke$Mars_tune <- predict(stroke_marsTune, stroke_testxpr)
testResults_stroke$SVMRadial <- predict(stroke_svmRTune, stroke_testxpr)
testResults_stroke$SVMPoly <- predict(stroke_svmPTune, stroke_testxpr)
testResults_stroke$KNN <- predict(stroke_knnTune, stroke_testxpr)

table(testResults_stroke$OBS == testResults_stroke$LDA)
table(testResults_stroke$OBS == testResults_stroke$GLMN)
table(testResults_stroke$OBS == testResults_stroke$NSC)
table(testResults_stroke$OBS == testResults_stroke$Mars_tune)
table(testResults_stroke$OBS == testResults_stroke$SVMRadial)
table(testResults_stroke$OBS == testResults_stroke$SVMPoly)
table(testResults_stroke$OBS == testResults_stroke$KNN)

trainup_stroke<-upSample(x=stroke_train[,-ncol(stroke_train)],
                         y=stroke_train$stroke)

table(stroke_train$stroke)
stroke_train$stroke <- as.numeric(as.factor(stroke_train$stroke))
table(stroke_train$stroke)

library(DMwR2)
trainsmote <- SMOTE(stroke~.,data = stroke_train)



