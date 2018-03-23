library(purrr)

splitData <- function(df, percent) {
  ## 75% of the sample size
  smp_size <- floor(percent * nrow(df))
  
  ## set the seed to make your partition reproductible
  set.seed(123)
  train_ind <- sample(seq_len(nrow(df)), size = smp_size)
  
  train_df <- df[train_ind, ]
  test_df <- df[-train_ind, ]
  return(list("train" = data.matrix(train_df), "test" = data.matrix(test_df)))
}


setwd("C:/Users/Rum/OneDrive/School/Boise State University/CS354/Project")
data = read.csv("carCrashData.csv")

x_data = subset(data, select=c("dvcat","weight","airbag","seatbelt","frontal","sex","ageOFocc","yearacc","yearVeh","abcat","occRole","deploy","injSeverity"))
y_data = subset(data, select=c("dead"))
ncol(y_data)


x_data = splitData(x_data, 0.8)
y_data = splitData(y_data, 0.8)

#nrow(x_data$train)
#nrow(x_data$test)
#length(y_data$train)
#length(y_data$test)

#print(x_data$train[978,])
#print(y_data$train[978])

print(class(y_data$train))
y_data$train <- to_categorical(y_data$train, 2)
y_data$test <- to_categorical(y_data$test, 2)
print(y_data$train)

#model
model <- keras_model_sequential()
model %>%
  layer_dense(units = 1000, activation = 'relu', input_shape = c(ncol(x_data$train))) %>%
  #layer_dropout(rate = 0.4) %>%
  #layer_dense(units = 128, activation = 'relu') %>%
  #layer_dropout(rate = 0.3) %>%
  layer_dense(units = 2, activation = 'softmax')

summary(model)

#compile
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

#train and eval
history <- model %>% fit(
  x_data$train, y_data$train,
  epochs = 30, batch_size = 128,
  validation_split = 0.2
)

plot(history)

model %>% evaluate(x_data$test, y_data$test)

y_prediction = model %>% predict_classes(x_data$test)




