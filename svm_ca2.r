
#install.packages("ISLR")
library(ISLR)
#install.packages("e1071")
library(e1071)
#install.packages("caret")
library(caret)
#install.packages("caTools")
library(caTools)
data(Default)
summary(Default)

set.seed(111)
split = sample.split(Default$default, SplitRatio = 0.8)

train = subset(Default, split == T)
test = subset(Default, split == F)

# train and test sets with balance and income only
train_bi = train[, c('default', 'balance', 'income')]
test_bi = test[, c('default', 'balance', 'income')]

# train and test sets with student and balance only
train_bs = train[, c('default', 'balance', 'student')]
test_bs = test[, c('default', 'balance', 'student')]

# train and test sets with student and balance only
train_is = train[, c('default', 'income', 'student')]
test_is = test[, c('default', 'income', 'student')]

svm_bi1 = svm(default ~ .,
           data = train_bi,
           kernel = 'radial',
           gamma = 1,
           cost = 1)

summary(svm_bi1)

set.seed(111)

tune_out_bi1 = tune(svm,
                 default ~ .,
                 data = train_bi,
                 kernel = 'radial',
                 ranges = list(cost = c(0.1, 1, 10, 100, 1000),
                               gamma = c(0.5, 1, 2, 3, 4)),
                 tunecontrol = tune.control(sampling = 'cross',
                                            cross = 5))

summary(tune_out_bi1)

bestmod_bi1 = tune_out_bi1$best.model
summary(bestmod_bi1)

# CM on Test Set
newpred_bi_test1 = predict(bestmod_bi1, test_bi)
confusionMatrix(table(prediction = newpred_bi_test1,
                      actual = test_bi$default))

svm_bi2 = svm(default ~ .,
           data = train_bi,
           kernel = 'sigmoid',
           gamma = 1,
           cost = 1)

summary(svm_bi2)

set.seed(111)

tune_out_bi2 = tune(svm,
                 default ~ .,
                 data = train_bi,
                 kernel = 'sigmoid',
                 ranges = list(cost = c(0.1, 1, 10, 100, 1000),
                               gamma = c(0.5, 1, 2, 3, 4)),
                 tunecontrol = tune.control(sampling = 'cross',
                                            cross = 5))

summary(tune_out_bi2)

bestmod_bi2 = tune_out_bi2$best.model
summary(bestmod_bi2)

# CM on Test Set
newpred_bi_test2 = predict(bestmod_bi2, test_bi)
confusionMatrix(table(prediction = newpred_bi_test2,
                      actual = test_bi$default))

svm_bs1 = svm(default ~ .,
           data = train_bs,
           kernel = 'radial',
           gamma = 1,
           cost = 1)

summary(svm_bs1)

set.seed(111)

tune_out_bs1 = tune(svm,
                 default ~ .,
                 data = train_bs,
                 kernel = 'radial',
                 ranges = list(cost = c(0.1, 1, 10, 100, 1000),
                               gamma = c(0.5, 1, 2, 3, 4)),
                 tunecontrol = tune.control(sampling = 'cross',
                                            cross = 5))

summary(tune_out_bs1)

bestmod_bs1 = tune_out_bs1$best.model
summary(bestmod_bs1)

# CM on Test Set
newpred_bs_test1 = predict(bestmod_bs1, test_bs)
confusionMatrix(table(prediction = newpred_bs_test1,
                      actual = test_bs$default))

svm_bs2 = svm(default ~ .,
           data = train_bs,
           kernel = 'sigmoid',
           gamma = 1,
           cost = 1)

summary(svm_bs2)

set.seed(111)

tune_out_bs2 = tune(svm,
                 default ~ .,
                 data = train_bs,
                 kernel = 'sigmoid',
                 ranges = list(cost = c(0.1, 1, 10, 100, 1000),
                               gamma = c(0.5, 1, 2, 3, 4)),
                 tunecontrol = tune.control(sampling = 'cross',
                                            cross = 5))

summary(tune_out_bs2)

bestmod_bs2 = tune_out_bs2$best.model
summary(bestmod_bs2)

# CM on Test Set
newpred_bs_test2 = predict(bestmod_bs2, test_bs)
confusionMatrix(table(prediction = newpred_bs_test2,
                      actual = test_bs$default))

svm_is1 = svm(default ~ .,
           data = train_is,
           kernel = 'radial',
           gamma = 1,
           cost = 1)

summary(svm_is1)

set.seed(111)

tune_out_is1 = tune(svm,
                 default ~ .,
                 data = train_is,
                 kernel = 'radial',
                 ranges = list(cost = c(0.1, 1, 10, 100, 1000),
                               gamma = c(0.5, 1, 2, 3, 4)),
                 tunecontrol = tune.control(sampling = 'cross',
                                            cross = 5))

summary(tune_out_is1)

bestmod_is1 = tune_out_is1$best.model
summary(bestmod_is1)

# CM on Test Set
newpred_is_test1 = predict(bestmod_is1, test_is)
confusionMatrix(table(prediction = newpred_is_test1,
                      actual = test_is$default))

svm_is2 = svm(default ~ .,
           data = train_is,
           kernel = 'sigmoid',
           gamma = 1,
           cost = 1)

summary(svm_is2)

set.seed(111)

tune_out_is2 = tune(svm,
                 default ~ .,
                 data = train_is,
                 kernel = 'sigmoid',
                 ranges = list(cost = c(0.1, 1, 10, 100, 1000),
                               gamma = c(0.5, 1, 2, 3, 4)),
                 tunecontrol = tune.control(sampling = 'cross',
                                            cross = 5))

summary(tune_out_is2)

bestmod_is2 = tune_out_is2$best.model
summary(bestmod_is2)

# CM on Test Set
newpred_is_test2 = predict(bestmod_is2, test_is)
confusionMatrix(table(prediction = newpred_is_test2,
                      actual = test_is$default))

svm_all1 = svm(default ~ .,
           data = train,
           kernel = 'radial',
           gamma = 1,
           cost = 1)

summary(svm_all1)

set.seed(111)

tune_out_all1 = tune(svm,
                 default ~ .,
                 data = train,
                 kernel = 'radial',
                 ranges = list(cost = c(0.1, 1, 10, 100, 1000),
                               gamma = c(0.5, 1, 2, 3, 4)),
                 tunecontrol = tune.control(sampling = 'cross',
                                            cross = 5))

summary(tune_out_all1)

bestmod_all1 = tune_out_all1$best.model
summary(bestmod_all1)

# CM on Test Set
newpred_all_test1 = predict(bestmod_all1, test)
confusionMatrix(table(prediction = newpred_all_test1,
                      actual = test$default))

svm_all2 = svm(default ~ .,
           data = train,
           kernel = 'sigmoid',
           gamma = 1,
           cost = 1)

summary(svm_all2)

set.seed(111)

tune_out_all2 = tune(svm,
                 default ~ .,
                 data = train,
                 kernel = 'sigmoid',
                 ranges = list(cost = c(0.1, 1, 10, 100, 1000),
                               gamma = c(0.5, 1, 2, 3, 4)),
                 tunecontrol = tune.control(sampling = 'cross',
                                            cross = 5))

summary(tune_out_all2)

bestmod_all2 = tune_out_all2$best.model
summary(bestmod_all2)

# CM on Test Set
newpred_all_test2 = predict(bestmod_all2, test)
confusionMatrix(table(prediction = newpred_all_test2,
                      actual = test$default))
