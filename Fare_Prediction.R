#Load Libraries
x = c("ggplot2", "corrgram", "DMwR", "tidyverse", "lubridate", "geosphere", "caret", "mice", "car", "dataPreparation","corrplot","randomForest","xgboost")

#install.packages(x)
lapply(x, require, character.only = TRUE)
rm(x)

#Load the data
train_df = read.csv("train_cab/train_cab.csv", header = TRUE, na.strings = c(" ", "", "NA"))
test_df = read.csv("test/test.csv", header = TRUE, na.strings = c(" ", "", "NA"))

str(train_df)
str(test_df)

# Exploratory Data Analysis
train_df$fare_amount = as.numeric(as.character(train_df$fare_amount))
train_df$passenger_count = round(train_df$passenger_count)

test_df$passenger_count = round(test_df$passenger_count)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Feature Extraction/Engineering~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Extract features from pickup_datetime
train_df$pickup_datetime <- parse_datetime(as.character(train_df$pickup_datetime), '%Y-%m-%d %H:%M:%S %Z')
train_df <- subset(train_df, !is.na(train_df$pickup_datetime)) #Omit the na time

train_df <- train_df %>%
    mutate(year = year(pickup_datetime)) %>%
    mutate(month = month(pickup_datetime)) %>%
    mutate(day = day(pickup_datetime)) %>%
    mutate(weekday = weekdays(pickup_datetime)) %>%
    mutate(hour = hour(pickup_datetime)) %>%
    mutate(minute = minute(pickup_datetime)) %>%
    mutate(second = second(pickup_datetime)) %>%
    select(-c(pickup_datetime))

train_df$weekday = as.factor(train_df$weekday)

test_df$pickup_datetime <- parse_datetime(as.character(test_df$pickup_datetime), '%Y-%m-%d %H:%M:%S %Z')

test_df <- test_df %>%
    mutate(year = year(pickup_datetime)) %>%
    mutate(month = month(pickup_datetime)) %>%
    mutate(day = day(pickup_datetime)) %>%
    mutate(weekday = weekdays(pickup_datetime)) %>%
    mutate(hour = hour(pickup_datetime)) %>%
    mutate(minute = minute(pickup_datetime)) %>%
    mutate(second = second(pickup_datetime)) %>%
    select(-c(pickup_datetime))

test_df$weekday = as.factor(test_df$weekday)

#Removing the invalid data
train_df <- train_df[(train_df$dropoff_latitude < 90) & (train_df$dropoff_latitude > -90),]
train_df <- train_df[(train_df$pickup_latitude < 90) & (train_df$pickup_latitude > -90),]
train_df <- train_df[(train_df$dropoff_longitude < 180) & (train_df$dropoff_longitude > -180),]
train_df <- train_df[(train_df$pickup_longitude < 180) & (train_df$pickup_longitude > -180),]

train_df <- train_df[(train_df$dropoff_latitude != train_df$pickup_latitude) & (train_df$dropoff_longitude != train_df$pickup_longitude),]
#train_df <- train_df[(train_df$dropoff_longitude != train_df$pickup_longitude)]

test_df <- test_df[(test_df$dropoff_latitude < 90) & (test_df$dropoff_latitude > -90),]
test_df <- test_df[(test_df$pickup_latitude < 90) & (test_df$pickup_latitude > -90),]
test_df <- test_df[(test_df$dropoff_longitude < 180) & (test_df$dropoff_longitude > -180),]
test_df <- test_df[(test_df$pickup_longitude < 180) & (test_df$pickup_longitude > -180),]

test_df <- test_df[(test_df$dropoff_latitude != test_df$pickup_latitude) & (test_df$dropoff_longitude != test_df$pickup_longitude),]

#Extract the distance from the geospatial coordinates
train_df <- train_df %>%
    mutate(distance = pmap_dbl(., ~ distm(c(..2, ..3), c(..4, ..5)) / 1000)) %>%
    select(-c(pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude))

test_df <- test_df %>%
    mutate(distance = pmap_dbl(., ~ distm(c(..1, ..2), c(..3, ..4)) / 1000)) %>%
    select(-c(pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude))

#~~~~~~~~~~~~~~~~~~~~~~~~~Missing value Analysis~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
for (i in names(train_df)) {
    print(paste("The number of NAs in ", i, "is", sum(is.na(train_df[[i]]))))   
}

for (i in names(test_df)) {
    print(paste("The number of NAs in ", i, "is", sum(is.na(test_df[[i]]))))
}
#No missing values in test datset
train_df <- subset(train_df, !is.na(train_df$fare_amount))
train_df <- subset(train_df, !is.na(train_df$passenger_count))
#print(paste("Variance in passenger_count without NAs is = ", var(train_df$passenger_count, na.rm = "TRUE"))) #3702.456
#print(paste("Variance in fare_amount without NAs is = ", var(train_df$fare_amount, na.rm = "TRUE"))) #185319.7

##After trying out various techniques like mean, median, we decided to go with sd as it tends to preserve the variance of the data
#print(paste("Variance in passenger_count with SD imputation is = ", var(with(train_df, impute(passenger_count, "random")))))
#train_df$passenger_count <- with(train_df, impute(passenger_count, "random"))

#print(paste("Variance in fare_amount with SD imputation is = ", var(with(train_df, impute(fare_amount, "random")))))
#train_df$fare_amount <- with(train_df, impute(fare_amount, "random"))

#####################                        Outlier Analysis                 ##################
numeric_index = sapply(train_df, is.numeric)
numeric_data = train_df[, numeric_index]
cnames = colnames(numeric_data)

for (i in 1:length(cnames))
{
        assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "fare_amount"), data = subset(train_df))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
        geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,outlier.size=1, notch=FALSE) +
        theme(legend.position="bottom")+
        labs(y=cnames[i],x="fare_amount")+
        ggtitle(paste("Box plot of fare_amount for",cnames[i])))
}
## Plotting plots together
gridExtra::grid.arrange(gn1, gn2, gn9, ncol = 3)
gridExtra::grid.arrange(gn3, gn4, gn5, ncol = 3)
gridExtra::grid.arrange(gn6, gn7, gn8, ncol = 3)

train_df_before_outliers <- train_df
#train_df$fare_amount[train_df$fare_amount %in% boxplot.stats(train_df$fare_amount)$out]
quantiles.fare_Amount <- quantile(train_df$fare_amount, c(.01, .99)) # 3.3  53.256
outliers_fare_amount = train_df[train_df$fare_amount < quantiles.fare_Amount[[1]] | train_df$fare_amount > quantiles.fare_Amount[[2]],]
train_df[, "fare_amount"][train_df[, "fare_amount"] %in% outliers_fare_amount$fare_amount] = NA

quantiles.distance <- quantile(train_df$distance, c(.01, .99)) # 0.2353299 20.8384309
outliers_distance = train_df[train_df$distance < quantiles.distance[[1]] | train_df$distance > quantiles.distance[[2]],]
train_df[, "distance"][train_df[, "distance"] %in% outliers_distance$distance] = NA

quantiles.passenger_count <- quantile(train_df$passenger_count, c(.02, .98)) # 1 5.56
outliers_passenger_count = train_df[train_df$passenger_count < quantiles.passenger_count[[1]] | train_df$passenger_count > quantiles.passenger_count[[2]],]
train_df[, "passenger_count"][train_df[, "passenger_count"] %in% outliers_passenger_count$passenger_count] = NA

##Tried various imputation techniques like mice,amelia,hmisc,mi but the impute function within mice is the one which preserves the variance and ideal for regression
#With mice
mice_imputed <- mice(train_df, m = 2, maxit = 50, method = 'pmm', seed = 500)
train_df.imputed <- complete(mice_imputed, 1)
train_df <- train_df.imputed

#With Hmisc
#train_df$fare_amount <- with(train_df, impute(fare_amount, "random"))
#train_df$distance <- with(train_df, impute(distance, "random"))
#train_df$passenger_count <- with(train_df, impute(passenger_count, "random"))

##################################Feature Selection################################################
## Correlation Plot
cor <- cor(numeric_data, method = "spearman")

#corrgram(numeric_data, order = F,
#upper.panel = panel.pie, text.panel = panel.txt, main = "Correlation Plot")

corrplot(cor, method = "circle")

#ANOVA for categorical variables with target numeric variable
aov_results = aov(fare_amount ~ weekday, data = train_df)
summary(aov_results)
print(paste("p value of ANOVA test",summary(aov_results)[[1]][["Pr(>F)"]]))
#Pr(>F) = 0.0728 So we reject the null hypothesis and drop the variable weekday into regression

#From the above tests we come to know that fare_amount is greatly influenced by distance and then year. Also weekday influences the change in fare_amount
## Dimension Reduction
df_train <- train_df
train_df = subset(train_df, select = c(fare_amount, distance, year))

#################### Splitting training data into train and validation subsets ###################
set.seed(1000)
tr.idx = createDataPartition(train_df$fare_amount, p = .7, list = FALSE)
train_data = train_df[tr.idx,]
test_data = train_df[-tr.idx,]

######################### Feature Scaling #######################################
scales <- build_scales(train_df, verbose = TRUE)
train_data.scale <- fastScale(train_data, scales = scales)
test_data.scale <- fastScale(test_data, scales = scales)

#Encoding of the categorical variable
#encoding <- build_encoding(dataSet = train_data.scale, cols = "auto", verbose = TRUE)
#train_data <- one_hot_encoder(dataSet = train_data.scale, encoding = encoding, drop = TRUE, verbose = TRUE)
#test_data <- one_hot_encoder(dataSet = test_data.scale, encoding = encoding, drop = TRUE, verbose = TRUE)
###################Model Selection################
#############            Linear regression               #################
lm_model = lm(fare_amount ~ ., data = train_data.scale)

summary(lm_model) #Multiple R-squared:  0.8314,	Adjusted R-squared:  0.8314

#Check for multicollinearity
vif(lm_model)

#Check for auto correlation of the residuals
durbinWatsonTest(lm_model) #1.967761

#vif values are near to 1 so there is no multicollinearity

plot(lm_model$fitted.values, rstandard(lm_model), main = "Residual plot",
     xlab = "Predicted values of fare_amount",
     ylab = "standardized residuals")


lm_predictions = predict(lm_model, test_data.scale)

predicted_data <- test_data.scale #perform a copy of test_data
predicted_data$fare_amount <- lm_predictions #Replace the predictions on the fare_amount

predicted_data <- fastScale(predicted_data, scales = scales, way = "unscale", verbose = TRUE) #Inverse the scale transform

qplot(x = test_data$fare_amount, y = predicted_data$fare_amount, data = test_data, color = I("blue"), geom = "point")
regr.eval(test_data$fare_amount, predicted_data$fare_amount)
#mae mse rmse mape
#2.0249371 11.2481155  3.3538210  0.1959903  
df <- cbind(test_data$distance, test_data$year, test_data$fare_amount, lm_predictions$Predicted_fare_amount) 

#Find R-Squared metrics
rsq <- function(x, y) summary(lm(y ~ x))$r.squared
rsq(test_data$fare_amount, predicted_data$fare_amount)
#0.8252926

#############                             Decision Tree            #####################
#Let us use all the features in the Tree, split the data into test and train
set.seed(1000)
tr.idx = createDataPartition(df_train$fare_amount, p = .7, list = FALSE)
train_data = df_train[tr.idx,]
test_data = df_train[-tr.idx,]

Dt_model = rpart(fare_amount ~ ., data = train_data)

summary(Dt_model)
#Predict for new test cases
predictions_DT = predict(Dt_model, test_data)

qplot(x = test_data$fare_amount, y = predictions_DT, data = test_data, color = I("blue"), geom = "point")

regr.eval(test_data$fare_amount, predictions_DT)
#mae mse rmse mape
#2.3646192 14.4211279  3.7975160  0.2289304 

df_dtree <- cbind(test_data$distance, test_data$year, test_data$fare_amount, predictions_DT)
rsq(test_data$fare_amount, predictions_DT)
# 0.7931986

#############                             Random forest            #####################
rf_model = randomForest(fare_amount ~ ., data = train_data)

summary(rf_model)

rf_predictions = predict(rf_model, test_data)

qplot(x = test_data$fare_amount, y = rf_predictions, data = test_data, color = I("blue"), geom = "point")

regr.eval(test_data$fare_amount, rf_predictions)
#mae          mse       rmse      mape
#1.9720581 11.0637578 3.3262228 0.1965603

rsq(test_data$fare_amount, rf_predictions)
#0.8353143

df_rf <- cbind(test_data$distance, test_data$year, test_data$fare_amount, rf_predictions)

############ XGBOOST ###########################
train_data_matrix = as.matrix(sapply(train_data[-1], as.numeric))
test_data_data_matrix = as.matrix(sapply(test_data[-1], as.numeric))

xgboost_model = xgboost(data = train_data_matrix, label = train_data$fare_amount, nrounds = 15, verbose = FALSE)

summary(xgboost_model)
xgb_predictions = predict(xgboost_model, test_data_data_matrix)

qplot(x = test_data[, 1], y = xgb_predictions, data = test_data, color = I("blue"), geom = "point")

regr.eval(test_data[, 1], xgb_predictions)
#mae       mse      rmse     mape
#1.901361 11.455126 3.384542 0.181079

rsq(test_data$fare_amount, xgb_predictions)
#0.8302861

df_xgb <- cbind(test_data$distance, test_data$year, test_data$fare_amount, xgb_predictions)

################ Saving the model ############################
saveRDS(lm_model, "./final_linear_model_using_R.rds")
saveRDS(rf_model, "./final_model_using_RF.rds")
saveRDS(xgboost_model, "./final_model_using_xgb.rds")

# loading the saved model - Since the linear model gives better forecast for increasing distance we select linear model as winning model
winning_model <- readRDS("./final_linear_model_using_R.rds")
print(winning_model)

test_df_scale <- fastScale(test_df, scales = scales, way = "scale", verbose = TRUE)

# Predict on scoring dataset
test_predictions_lin = predict(winning_model, test_df_scale)

test_df.copy <- test_df_scale #create a copy
test_df.copy$fare_amount <- test_predictions_lin

test_df.copy <- fastScale(test_df.copy, scales = scales, way = "unscale", verbose = TRUE) #unscale the required variables from the dataframe
test_df$fare_amount <- test_df.copy$fare_amount

# Save the predicted fare_amount in csv format
write.csv(test_df, "test/test_predictions_R_lm.csv", row.names = FALSE)