############################################################
####                   Mid term assignment              ####
####                    by Nihel Charfi                 ####
############################################################

# To clean up the memory of your current R session run the following line
rm(list=ls(all=TRUE))

#Install packages and libraries
install.packages("corrplot")
install.packages("gridExtra")
install.packages("leaps")
install.packages("tibble")
install.packages("caret")
library("tibble")
library(tidyverse)
library(lattice)
library(caret)
library(mlbench)
library(corrplot)
library(leaps)

# Establish the working directory
setwd("D:/Nihel/Marquette university/Spring 2019/Business Analytics/Mid term Project")
bc_data<- read.csv("breast-cancer-wisconsin-data.csv")
bc_data<- as.tibble(bc_data)
head(bc_data)
str(bc_data)

# how many NAs are in the data
sum(is.na(bc_data))
#total missing values in each column 
colSums(is.na(bc_data))
#Move diagnosis to the end of data frame
bc_data <- bc_data %>%
  select(-diagnosis, everything())

# impute missing data
library(mice)
bc_data[,1:32] <- apply(bc_data[,1:32], 2, function(x) as.numeric(as.character(x)))
dataset_impute <- mice(bc_data[,1:32],  print = FALSE)
bc_data<- cbind(bc_data[, 33, drop = FALSE], mice::complete(dataset_impute, 1))

#Remove variable X as it is not a dataset variable
bc_data$X  = NULL;
#Check again for the total missing values in each column
colSums(is.na(bc_data))

#Find thecorrelation between variables
cor_1<-cor(bc_data[-1],method="pearson")
corrplot(cor_1,method="number",type="lower")
corrplot(cor_1,type="lower")

#Filter out the columns having correlation value greater than 0.9
highly_correlated<-findCorrelation(cor_1,cutoff=0.9)
print(highly_correlated)
exclude<-colnames(cor_1)[highly_correlated]

# get the new breast cancer data after reducing the number of predictors
bc_data_new<-bc_data[,setdiff(colnames(bc_data), exclude)]
bc_data_new <- as.tibble(bc_data_new)
head(bc_data_new)

# Save the new data in csv file 
write.csv(bc_data_new,"bc_data_new.csv")

#Move diagnosis to the end of data frame
bcdata_new <- bc_data_new %>%
  select(id, everything())

# Jitter plot of the remaining variables
var_list<-names(bcdata_new)
library("gridExtra")

for (i in 2:length(var_list)) {
  gs <- lapply(3:22, function(i) 
    ggplot(data=bcdata_new,aes_string(x=var_list[[2]], y=var_list[[i]])) +
      geom_jitter(mapping = aes(color=diagnosis)))
}
grid.arrange(grobs=gs, ncol=5)

# Box plot of the variables to see the distribution of data
for (i in 2:length(var_list)) {
  gs <- lapply(3:22, function(i) 
    ggplot(data = bcdata_new,aes_string(x=var_list[[2]], y=var_list[[i]])) +
      geom_boxplot(mapping = aes(color=diagnosis)))
  
}
grid.arrange(grobs=gs, ncol=5)

#Partitioning into training and test sets
#Use set.seed()
set.seed(123)
trainIndex <- createDataPartition(bcdata_new$diagnosis, p = 0.8, list = FALSE, times = 1)
bcdata_Train <- bcdata_new[trainIndex, ]
bcdata_Test <- bcdata_new[-trainIndex, ]

#calculate the pre-process parameters from the dataset
scaler <- preProcess(bcdata_Train, method = c("center", "scale"))
bcdata_Train <- predict(scaler, bcdata_Train)
bcdata_Test <- predict(scaler, bcdata_Test)
head(bcdata_Train)
str(bcdata_Train)
bcdata_Train <- select(bcdata_Train,
                       area_mean, 
                       smoothness_mean,
                       compactness_mean,
                       symmetry_mean,
                       concave.points_se,
                       texture_worst,
                       smoothness_worst,
                       concave.points_worst,
                       symmetry_worst,
                       fractal_dimension_worst,
                       diagnosis)
head(bcdata_Train)

# install package e1071
install.packages('e1071', dependencies=TRUE)
library("e1071")

#Trains the model with all features 
#diagnosis  area_mean
knn_area_mean <- train(diagnosis ~ area_mean, data = bcdata_Train, method = "knn")
TestPredictions_area_mean <- predict(knn_area_mean, bcdata_Test)
confusionMatrix(TestPredictions_area_mean, bcdata_Test$diagnosis)

#diagnosis ~ smoothness_mean
knn_smoothness_mean <- train(diagnosis ~ smoothness_mean, data = bcdata_Train, method = "knn")
TestPredictions_smoothness_mean <- predict(knn_smoothness_mean, bcdata_Test)
confusionMatrix(TestPredictions_smoothness_mean, bcdata_Test$diagnosis)

#diagnosis ~ compactness_mean
knn_compactness_mean <- train(diagnosis ~ compactness_mean, data = bcdata_Train, method = "knn")
TestPredictions_compactness_mean <- predict(knn_compactness_mean, bcdata_Test)
confusionMatrix(TestPredictions_compactness_mean, bcdata_Test$diagnosis)

#diagnosis ~ symmetry_mean
knn_symmetry_mean <- train(diagnosis ~ symmetry_mean, data = bcdata_Train, method = "knn")
TestPredictions_symmetry_mean <- predict(knn_symmetry_mean, bcdata_Test)
confusionMatrix(TestPredictions_symmetry_mean, bcdata_Test$diagnosis)

#diagnosis ~ concave.points_se
knn_concave.points_se <- train(diagnosis ~ concave.points_se, data = bcdata_Train, method = "knn")
TestPredictions_concave.points_se <- predict(knn_concave.points_se, bcdata_Test)
confusionMatrix(TestPredictions_concave.points_se, bcdata_Test$diagnosis)

#diagnosis ~ texture_worst
knn_texture_worst <- train(diagnosis ~ texture_worst, data = bcdata_Train, method = "knn")
TestPredictions_texture_worst <- predict(knn_texture_worst, bcdata_Test)
confusionMatrix(TestPredictions_texture_worst, bcdata_Test$diagnosis)

#diagnosis ~ smoothness_worst
knn_smoothness_worst <- train(diagnosis ~ smoothness_worst, data = bcdata_Train, method = "knn")
TestPredictions_smoothness_worst <- predict(knn_smoothness_worst, bcdata_Test)
confusionMatrix(TestPredictions_smoothness_worst, bcdata_Test$diagnosis)

#diagnosis ~ concave.points_worst
knn_concave.points_worst <- train(diagnosis ~ concave.points_worst, data = bcdata_Train, method = "knn")
TestPredictions_concave.points_worst <- predict(knn_concave.points_worst, bcdata_Test)
confusionMatrix(TestPredictions_concave.points_worst, bcdata_Test$diagnosis)

#diagnosis ~ symmetry_worst
knn_symmetry_worst <- train(diagnosis ~ symmetry_worst, data = bcdata_Train, method = "knn")
TestPredictions_symmetry_worst <- predict(knn_symmetry_worst, bcdata_Test)
confusionMatrix(TestPredictions_symmetry_worst, bcdata_Test$diagnosis)

#diagnosis ~ fractal_dimension_worst
knn_fractal_dimension_worst <- train(diagnosis ~ fractal_dimension_worst, data = bcdata_Train, method = "knn")
TestPredictions_fractal_dimension_worst <- predict(knn_fractal_dimension_worst, bcdata_Test)
confusionMatrix(TestPredictions_fractal_dimension_worst, bcdata_Test$diagnosis)

#Round 2
#diagnosis ~ concave.points_worst + area_mean
knn_concave.points_worst_area_mean <- train(diagnosis ~ concave.points_worst + area_mean, data = bcdata_Train, method = "knn")
TestPredictions_concave.points_worst_area_mean  <- predict(knn_concave.points_worst_area_mean, bcdata_Test)
confusionMatrix(TestPredictions_concave.points_worst_area_mean , bcdata_Test$diagnosis)

#diagnosis ~ concave.points_worst + compactness_mean
knn_concave.points_worst_compactness_mean <- train(diagnosis ~ concave.points_worst + compactness_mean,
                                                   data = bcdata_Train, method = "knn")
TestPredictions_concave.points_worst_compactness_mean  <- predict(knn_concave.points_worst_compactness_mean, bcdata_Test)
confusionMatrix(TestPredictions_concave.points_worst_compactness_mean , bcdata_Test$diagnosis)

#diagnosis ~ concave.points_worst + texture_worst
knn_concave.points_worst_texture_worst <- train(diagnosis ~ concave.points_worst + texture_worst,
                                                   data = bcdata_Train, method = "knn")
TestPredictions_concave.points_worst_texture_worst  <- predict(knn_concave.points_worst_texture_worst, bcdata_Test)
confusionMatrix(TestPredictions_concave.points_worst_texture_worst, bcdata_Test$diagnosis)

#diagnosis ~ concave.points_worst + symmetry_worst
knn_concave.points_worst_symmetry_worst <- train(diagnosis ~ concave.points_worst + symmetry_worst,
                                                data = bcdata_Train, method = "knn")
TestPredictions_concave.points_worst_symmetry_worst<- predict(knn_concave.points_worst_symmetry_worst, bcdata_Test)
confusionMatrix(TestPredictions_concave.points_worst_symmetry_worst, bcdata_Test$diagnosis)

#Round 3
#diagnosis ~ concave.points_worst + area_mean + Compactness_mean
knn_concave.points_worst_area_mean_compactness_mean <- train(diagnosis ~ concave.points_worst + area_mean + compactness_mean,
                                                             data = bcdata_Train, method = "knn")
TestPredictions_concave.points_worst_area_mean_compactness_mean<- predict(knn_concave.points_worst_area_mean_compactness_mean, bcdata_Test)
confusionMatrix(TestPredictions_concave.points_worst_area_mean_compactness_mean, bcdata_Test$diagnosis)

#diagnosis ~ concave.points_worst + area_mean + texture_worst 
knn_concave.points_worst_area_mean_texture_worst <- train(diagnosis ~ concave.points_worst + area_mean + texture_worst,
                                                             data = bcdata_Train, method = "knn")
TestPredictions_concave.points_worst_area_mean_texture_worst<- predict(knn_concave.points_worst_area_mean_texture_worst, bcdata_Test)
confusionMatrix(TestPredictions_concave.points_worst_area_mean_texture_worst, bcdata_Test$diagnosis)

#diagnosis ~ concave.points_worst + area_mean + symmetry_worst 
knn_concave.points_worst_area_mean_symmetry_worst <- train(diagnosis ~ concave.points_worst + area_mean + symmetry_worst,
                                                          data = bcdata_Train, method = "knn")
TestPredictions_concave.points_worst_area_mean_symmetry_worst<- predict(knn_concave.points_worst_area_mean_symmetry_worst, bcdata_Test)
confusionMatrix(TestPredictions_concave.points_worst_area_mean_symmetry_worst, bcdata_Test$diagnosis)

#Round 4
#diagnosis ~ concave.points_worst + area_mean + symmetry_worst +compactness_mean
knn_concave.points_worst_area_mean_symmetry_worst_compactness_mean <- train(diagnosis ~ concave.points_worst +
                                                             area_mean + 
                                                             symmetry_worst +
                                                             compactness_mean,
                                                           data = bcdata_Train, method = "knn")
TestPredictions_concave.points_worst_area_mean_symmetry_worst_compactness_mean <- predict(
                knn_concave.points_worst_area_mean_symmetry_worst_compactness_mean, bcdata_Test)
confusionMatrix(TestPredictions_concave.points_worst_area_mean_symmetry_worst_compactness_mean, bcdata_Test$diagnosis)

#diagnosis ~ concave.points_worst + area_mean + symmetry_worst + texture_worst
knn_concave.points_worst_area_mean_symmetry_worst_texture_worst <- train(diagnosis ~ concave.points_worst +
                                                                              area_mean + 
                                                                              symmetry_worst +
                                                                              texture_worst,
                                                                            data = bcdata_Train, method = "knn")
TestPredictions_concave.points_worst_area_mean_symmetry_worst_texture_worst <- predict(
                             knn_concave.points_worst_area_mean_symmetry_worst_texture_worst, bcdata_Test)
confusionMatrix(TestPredictions_concave.points_worst_area_mean_symmetry_worst_texture_worst, bcdata_Test$diagnosis)

#Round 5
#diagnosis ~ concave.points_worst + area_mean + symmetry_worst + texture_worst + compactness_mean 
knn_concave.points_worst_area_mean_symmetry_worst_texture_worst_compactness_mean <- train(diagnosis ~ concave.points_worst +
                                                                           area_mean + 
                                                                           symmetry_worst +
                                                                           texture_worst +
                                                                          compactness_mean,
                                                                         data = bcdata_Train, method = "knn")
TestPredictions_concave.points_worst_area_mean_symmetry_worst_texture_worst_compactness_mean <- predict(
                          knn_concave.points_worst_area_mean_symmetry_worst_texture_worst_compactness_mean, bcdata_Test)
confusionMatrix(TestPredictions_concave.points_worst_area_mean_symmetry_worst_texture_worst_compactness_mean, bcdata_Test$diagnosis)