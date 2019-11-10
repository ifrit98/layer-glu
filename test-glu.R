library(tensorflow)
library(keras)
library(magrittr)
source('layer-glu.R')


# Data Preparation --------------------------------------------------------

batch_size <- 128L
num_classes <- 10L
epochs <- 40L

# The data, shuffled and split between train and test sets
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# Redimension
x_train <- array_reshape(x_train, c(nrow(x_train), 784, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), 784, 1))

# Transform RGB values into [0,1] range
x_train <- x_train / 255
x_test <- x_test / 255

cat(nrow(x_train), 'train samples\n')
cat(nrow(x_test), 'test samples\n')

# Convert class vectors to binary class matrices
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)


# Helper function
build_and_compile <-
  function(input,
           output,
           optimizer = 'adam',
           loss      = "mse",
           metric    = 'acc') {

    model <-keras_model(input, output) %>%
      compile(optimizer = optimizer,
              loss      = loss,
              metric    = metric)
    model
  }


# Define Model -------------------------------------------------

make_glu_model <-
  function(input_shape = list(784L, 1L),
           output_classes = vector(length = num_classes)) {

    input <-
      layer_input(shape = input_shape)

    glu <- input %>%
      layer_glu(4, 3) %>%
      layer_max_pooling_1d() %>% 
      layer_glu(8, 5) %>%
      layer_max_pooling_1d() %>% 
      layer_glu(16, 8) %>% 
      layer_max_pooling_1d() %>% 
      layer_glu(32, 12)

    output <- glu %>% 
      layer_global_max_pooling_1d() %>%
      layer_dense(length(output_classes))

    build_and_compile(input, output)
  }


(model <- make_glu_model())

model$fit(
  x_train, y_train,
  epochs = epochs,
  batch_size = batch_size,
  validation_data = list(x_test, y_test)
)

