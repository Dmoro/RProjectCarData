#library(keras)
#install_keras()

mnist <-dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# reshape
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))

# rescale
x_train <- x_train / 255
x_test <- x_test / 255

#one-hot
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

class(x_train)

#model
model <- keras_model_sequential()
model %>%
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>%
  #layer_dropout(rate = 0.4) %>%
  #layer_dense(units = 128, activation = 'relu') %>%
  #layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')

summary(model)

#compile
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

#train and eval
history <- model %>% fit(
  x_train, y_train,
  epochs = 30, batch_size = 128,
  validation_split = 0.2
)

plot(history)

model %>% evaluate(x_test, y_test)

y_prediction = model %>% predict_classes(x_test)


e = round(runif(1,1,nrow(x_test)))
m = matrix(x_test[e,],28,28)
par(mar=c(0, 0, 0, 0))
m_rev <- apply(t(m), 2, rev)
image(t(m_rev), useRaster=TRUE, axes=FALSE)
message("guessed: ", y_prediction[e])
message("correct: ", which.max(y_test[e,])-1)
