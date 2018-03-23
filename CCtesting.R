library(purrr)

splitData <- function(df, percent) {
  ## 75% of the sample size
  smp_size <- floor(percent * nrow(df))
  
  ## set the seed to make your partition reproductible
  set.seed(123)
  train_ind <- sample(seq_len(nrow(df)), size = smp_size)
  
  train_df <-  data.frame(df[train_ind, ])
  test_df <- data.frame(df[-train_ind, ])
  
  return(list("train" = data.matrix(train_df), "test" = data.matrix(test_df)))
}


data = read.csv("carCrashData.csv")

x_data = subset(data, select=c("dvcat","weight","airbag","seatbelt","frontal","sex",
                               "ageOFocc","yearacc","yearVeh","abcat","occRole","deploy","injSeverity"))
y_data = subset(data, select=c("dead"))
ncol(y_data)

#split the data amont training and testing
x_data = splitData(x_data, 0.8)
y_data = splitData(y_data, 0.8) 

#convert the y data to a one dimensional array
y_data$train = as.vector(y_data$train)
y_data$test = as.vector(y_data$test)


#convert one dimensional array to a categorical matrix
y_data$train <- to_categorical(y_data$train, 3)
y_data$train <-y_data$train[,-1]
y_data$test <- to_categorical(y_data$test, 3)
y_data$test <-y_data$test[,-1]

print(y_data$train)
print(y_data$test)
print(x_data$train)
print(x_data$test)


#model
model <- keras_model_sequential()
model %>%
  layer_dense(units = 1000, activation = 'relu', input_shape = c(ncol(x_data$train))) %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
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
print(y_prediction)

print(x_data$train[1, ])
print(y_data$train[1, ])

B = matrix( 
  c( 2, 19.287, 2,1, 1,1,80,1998,1981,3,1,0,3), 
  nrow=1, 
  ncol=13) 
print(B)

print(model %>% predict_classes(B))

