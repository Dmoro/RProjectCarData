library(keras)
library(caret)
library(e1071)
library(plotly)
splitData <- function(df, percent, sample) {
  train_ind <- sample
  train_df <-  data.frame(df[train_ind, ])
  test_df <- data.frame(df[train_ind, ])
  
  return(list("train" = data.matrix(train_df), "test" = data.matrix(test_df)))
}

# Cleaned carCrashData to convert severity 5 to 0, 6 to 4, and NA's to 0.
data = read.csv("carCrashDataCleanedNEW.csv")
print(data)

x_data = subset(data, select=c("dvcat","dead", "airbag","seatbelt","frontal","sex",
                               "ageOFocc","abcat","occRole","deploy", 'ageVeh'))

#x_data = subset(data, select=c("abcat","airbag","seatbelt","dvcat","sex","ageVeh","occRole"))

y_data = subset(data, select=c("injSeverity"))
nrow(y_data) # equals 26,217 colums with row injSeverity

#split the data into list(matrix,matrix) with 70% training and 30% testing respectively.
set.seed(123)
percent = 0.7
smp_size <- floor(percent * nrow(x_data))
sample = sample(seq_len(nrow(x_data)), size = smp_size)
x_data = splitData(x_data, 0.7, sample)
y_data = splitData(y_data, 0.7, sample) 


#convert the y data to a one dimensional array
y_data$train = as.vector(y_data$train)
y_data$test = as.vector(y_data$test)

#convert one dimensional array to a categorical matrix via one-hot encoding
temp = y_data$test
print(y_data$train)
y_data$train <- to_categorical(y_data$train, 4)
y_data$test <- to_categorical(y_data$test, 4)



print(x_data$train[1:10,])
print(y_data$test[1:10,])

#model
model <- keras_model_sequential()
model %>%
  layer_dense(units = 50, activation = 'relu', input_shape = c(ncol(x_data$train)), kernel_initializer = "random_uniform") %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 50, activation = 'relu', kernel_initializer = "random_uniform") %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 50, activation = 'relu', kernel_initializer = "random_uniform") %>%
  layer_dropout(rate = 0.1) %>%
  #layer_dense(units = 128, activation = 'relu', kernel_initializer = "random_uniform") %>%
  #layer_dropout(rate = 0.1) %>%
  layer_dense(units = 4, activation = 'softmax') # increased our units to 5 to reflect all categories of injSeverity

summary(model)


#compile
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam', #optimizer_sgd(lr=0.01), # changed optimizer optimizer_rmsprop() to optimizer_sgd(lr0.0001)
  #optimizer = optimizer_rmsprop(),  # MNIST optimizer
  metrics = c('accuracy')
)

print(x_data$train[1:10])
print(x_data$train[1:10])

#train and eval
history <- model %>% fit(
  x_data$train, y_data$train,
  epochs = 50, batch_size = 100,
  validation_split = 0.2
)

plot(history)

model %>% evaluate(x_data$test, y_data$test)

y_prediction = model %>% predict_classes(x_data$test)
print(y_prediction[1:10000])
print(y_data$test)

cm = confusionMatrix(as.factor(y_prediction), as.factor(temp), positive = NULL, dnn =c("Prediction","Reference"))
print(cm$table)
print(x_data$train[1, ])
print(y_data$train[1, ])

m <- cm$table

for ( c in 1:ncol(m)) {
  c_sum = sum(m[,c])
  for (r in 1:length(m[,c])) {
    m[r,c] = m[r,c] / c_sum
  }
}

print(m)

p <- plot_ly(
  x = c("0", "1", "2","3"), y = c("0", "1", "2","3"),
  z = m, type = "heatmap" )
 #%>%
#  layout(
#title = "The Heatmap",
    
 #     xaxis = list(title = "reference"),
  #    yaxis = list(title = "prediction"),
    
  #  )
print(p)


numcorrect = 0
numtests = 1000
for(row in c(0: numtests)) {
  testnum = row
  #print(testnum)
  B = matrix (
    x_data$test[testnum,],
    nrow=1,
    ncol=ncol(x_data$test))
  #print(B)
  #print(c(guess, gold))
  guess = (model %>% predict_classes(B))
  gold = (which.max(y_data$test[testnum,])-1)
  
  if (identical(guess[1],gold[1])) { #|| identical(guess[1],gold[1]+1) || identical(guess[1],gold[1]-1)) {
    numcorrect = numcorrect + 1
  }
}

print(numcorrect)
print(numcorrect / numtests)



