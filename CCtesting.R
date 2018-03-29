library(keras)

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
#   Original Data Set
#data = read.csv("carCrashData.csv")

# Cleaned carCrashData to convert severity 5 to 0, 6 to 4, and NA's to 0.
data = read.csv("carCrashDataCleaned.csv")

#x_data = subset(data, select=c("dvcat","weight","airbag","seatbelt","frontal","sex",
#                              "ageOFocc","yearacc","yearVeh","abcat","occRole","deploy","injSeverity"))

x_data = subset(data, select=c("dvcat", "dead", "airbag","seatbelt","frontal","sex","abcat","occRole","deploy"))

y_data = subset(data, select=c("injSeverity"))
nrow(y_data) # equals 26,217 colums with row injSeverity

#split the data into list(matrix,matrix) with 80% training and 20% testing respectively.
x_data = splitData(x_data, 0.8)
y_data = splitData(y_data, 0.8) 


#convert the y data to a one dimensional array
y_data$train = as.vector(y_data$train)
y_data$test = as.vector(y_data$test)

#convert one dimensional array to a categorical matrix via one-hot encoding
y_data$train <- to_categorical(y_data$train, 5)
y_data$test <- to_categorical(y_data$test, 5)

#model
model <- keras_model_sequential()
model %>%
  layer_dense(units = 100, activation = 'relu', input_shape = c(ncol(x_data$train))) %>%
  #layer_dropout(rate = 0.4) %>%
  layer_dense(units = 100, activation = 'relu') %>%
  layer_dense(units = 100, activation = 'relu') %>%
  #layer_dropout(rate = 0.3) %>%
  layer_dense(units = 100, activation = 'relu') %>%
  layer_dense(units = 100, activation = 'relu') %>%
  layer_dense(units = 5, activation = 'softmax') # increased our units to 5 to reflect all categories of injSeverity

summary(model)


#compile
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_sgd(lr=0.01), # changed optimizer optimizer_rmsprop() to optimizer_sgd(lr0.0001)
  #optimizer = optimizer_rmsprop(),  # MNIST optimizer
  metrics = c('accuracy')
)

print(x_data)

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

# test case line 15
B = matrix (
  c(3, 89., 2,1, 1,1,80,3,1,0,3),
  nrow=1,
  ncol=9)
print(B)

print(model %>% predict_classes(B))

