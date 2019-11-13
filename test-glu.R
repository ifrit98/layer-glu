library(tensorflow)
library(keras)
library(magrittr)
# source('layer-glu.R')


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

## CURRENT ERRORS
# Error in py_call_impl(callable, dots$args, dots$keywords) : 
#   TypeError: in converted code:
#   relative to /home/home/.local/lib/python3.6/site-packages/tensorflow_core/python:
#   
#   keras/engine/data_adapter.py:307 slice_batch_indices
# first_k_indices = array_ops.slice(indices, [0], [num_in_full_batch])
# ops/array_ops.py:866 slice
# return gen_array_ops._slice(input_, begin, size, name=name)
# ops/gen_array_ops.py:9224 _slice
# "Slice", input=input, begin=begin, size=size, name=name)
# framework/op_def_library.py:536 _apply_op_helper
# repr(values), type(values).__name__, err))
# 
# TypeError: Expected int32 passed to parameter 'size' of op 'Slice', got 
# [59904.0] of type 'list' instead. Error: Expected int32, got 59904.0 of type 'float' instead. 


make_glu_model <- 
  function(input_shape = list(784L, 1L), 
           output_classes = vector(length = num_classes)) {
    
    input <- 
      layer_input(shape = input_shape)
    
    glu <- input %>% 
      layer_glu(32, 3) %>% 
      layer_max_pooling_1d() %>% 
      layer_glu(64, 5) %>%
      layer_max_pooling_1d() %>% 
      layer_glu(128, 8)
      # layer_glu(192, 12) %>%
      # layer_glu(256, 16)
    
    output <- layer_global_max_pooling_1d(glu) %>% 
      layer_dense(length(output_classes))
    
    build_and_compile(input, output)
  }



make_glu_block <- 
  function(input_shape = list(784L, 1L), 
           output_classes = vector(length = num_classes)) {
    
    input <- 
      layer_input(shape = input_shape) 
    
    base <- input %>% 
      layer_glu_block()
    
    output <- base %>% 
      layer_global_max_pooling_1d() %>% 
      layer_dense(length(output_classes), activation = 'softmax')
    
    build_and_compile(input, output)
  }


# (model <- make_glu_block())
# (model <- make_glu_model())
(model <- gated_linear_unit())

model %>% compile(
  optimizer = 'adagrad',
  loss = 'categorical_crossentropy',
  metrics = 'acc'
)

model$fit(
  x_train, y_train,
  epochs = epochs,
  batch_size = batch_size,
  validation_data = list(x_test, y_test)
)

