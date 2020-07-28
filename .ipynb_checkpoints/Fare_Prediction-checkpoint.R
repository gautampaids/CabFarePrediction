library(tidyverse)
library(lubridate)
library(mice)
library(geosphere)

train_df = read.csv("train_cab/train_cab.csv", header = TRUE)
test_df = read.csv("test/test.csv", header = TRUE)

str(train_df)
str(test_df)

# Exploratory Data Analysis
train_df$fare_amount = as.numeric(train_df$fare_amount)
train_df$passenger_count = round(train_df$passenger_count)

test_df$passenger_count = round(test_df$passenger_count)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Feature Extraction~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
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

#Removing the invalid data
train_df <- train_df[(train_df$dropoff_latitude < 90) | (train_df$dropoff_latitude > -90),]
train_df <- train_df[(train_df$pickup_latitude < 90) | (train_df$pickup_latitude > -90),]
train_df <- train_df[(train_df$dropoff_longitude < 180) | (train_df$dropoff_longitude > -180),]
train_df <- train_df[(train_df$pickup_longitude < 180) | (train_df$pickup_longitude > -180),]

test_df <- test_df[(test_df$dropoff_latitude < 90) | (test_df$dropoff_latitude > -90),]
test_df <- test_df[(test_df$pickup_latitude < 90) | (test_df$pickup_latitude > -90),]
test_df <- test_df[(test_df$dropoff_longitude < 180) | (test_df$dropoff_longitude > -180),]
test_df <- test_df[(test_df$pickup_longitude < 180) | (test_df$pickup_longitude > -180),]

#Extract the distance from the geospatial coordinates
train_df <- train_df %>%
    mutate(distance = distm(cbind(train_df$pickup_longitude, train_df$pickup_latitude), cbind(train_df$dropoff_longitude, train_df$dropoff_latitude))) %>%
    select(-c(pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude))

#Missing value Analysis
sum(is.na(train_df$passenger_count)) #There are 55 missing values in passenger_count
var(train_df$passenger_count, na.rm = "TRUE") #3702.225
imputed_Data <- mice(train_df, m = 1, maxit = 50, method = 'pmm', seed = 500)
completeData <- complete(imputed_Data, 1)
var(completeData$passenger_count) #3689

#Calculating distance
distm(c(train_df$pickup_longitude, train_df$pickup_latitude), c(train_df$dropoff_longitude, train_df$dropoff_latitude))