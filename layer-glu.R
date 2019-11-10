

GatedLinearUnit <-
  R6::R6Class(
    "GatedLinearUnit",
    
    inherit = KerasLayer,
    
    public = list(
      filters = NULL,
      kernel_size = NULL,
      kernel_initializer = NULL,
      kernel_regularizer = NULL,
      bias_initializer = NULL,
      bias_regularizer = NULL,
      linear_kernel = NULL,
      gated_kernel = NULL,
      linear_bias = NULL,
      gated_bias = NULL,
      kernel_shape = NULL,
      
      initialize = function(filters,
                            kernel_size,
                            kernel_initializer,
                            kernel_regularizer,
                            bias_initializer,
                            bias_regularizer) {
        self$filters <- filters
        self$kernel_size <- kernel_size
        self$kernel_initializer <- kernel_initializer
        self$kernel_regularizer <- kernel_regularizer
        self$bias_initializer <- bias_initializer
        self$bias_regularizer <- bias_regularizer
      },
      
      build = function(input_shape) {
        
        if (length(input_shape) == 1L)
          input_shape <- tf$TensorShape(input_shape[[1L]])
        
        self$kernel_shape <-
          list(self$kernel_size,
               input_shape[[length(input_shape)]],
               self$filters)
        
        self$linear_kernel <- self$add_weight(
          name = 'linear_kernel',
          shape = self$kernel_shape,
          initializer = self$kernel_initializer,
          regularizer = self$kernel_regularizer,
          trainable = TRUE
        )
        
        self$linear_bias <- self$add_weight(
          name = 'linear_bias',
          shape = self$filters,
          initializer = self$bias_initializer,
          regularizer = self$bias_regularizer,
          trainable = TRUE
        )
        
        self$gated_kernel <- self$add_weight(
          name = 'gated_kernel',
          shape = self$kernel_shape,
          initializer = self$kernel_initializer,
          regularizer = self$kernel_regularizer,
          trainable = TRUE
        )
        
        self$gated_bias <- self$add_weight(
          name = 'gated_bias',
          shape = self$filters,
          initializer = self$bias_initializer,
          regularizer = self$bias_regularizer,
          trainable = TRUE
        )
      },
      
      call = function(x, mask = NULL) {
        
        linear_out <-
          tf$keras$backend$conv1d(x, self$linear_kernel, padding = 'same') %>%
          tf$keras$backend$bias_add(self$linear_bias)
        
        gated_out <-
          tf$keras$backend$conv1d(x, self$gated_kernel, padding = 'same') %>%
          tf$keras$backend$bias_add(self$gated_bias)
        
        h <-
          layer_multiply(list(
            linear_out,
            layer_activation(gated_out, activation = 'sigmoid')
          ))
        
        h
      },
      
      compute_output_shape = function(input_shape) {
        list(NULL, input_shape[[2L]], self$filters)
      }
    )
  )


layer_glu <- 
  function(object,
           filters = 32,
           kernel_size = 3,
           kernel_initializer = 'glorot_normal',
           kernel_regularizer = NULL,
           bias_initializer = 'zeros',
           bias_regularizer = NULL,
           name = NULL,
           trainable = TRUE) {
    create_layer(
      GatedLinearUnit,
      object,
      list(
        filters = as.integer(filters),
        kernel_size = as.integer(kernel_size),
        kernel_initializer = tf$keras$initializers$get(kernel_initializer),
        kernel_regularizer = tf$keras$regularizers$get(kernel_regularizer),
        bias_initializer = tf$keras$initializers$get(bias_initializer),
        bias_regularizer = tf$keras$regularizers$get(bias_regularizer),
        name = name,
        trainable = trainable
      )
    )
  }




GLUBlock <-
  R6::R6Class(
    "GLUBlock",
    
    inherit = KerasLayer,
    
    public = list(
      filters = NULL,
      num_layers = NULL,
      kernel_size = NULL,
      kernel_initializer = NULL,
      kernel_regularizer = NULL,
      bias_initializer = NULL,
      bias_regularizer = NULL,
      glu_layers = NULL,
      
      initialize = function(num_layers,
                            filters,
                            kernel_size,
                            kernel_initializer,
                            kernel_regularizer,
                            bias_initializer,
                            bias_regularizer) {
        
        self$filters <- filters
        self$num_layers <- num_layers
        self$kernel_size <- kernel_size
        self$kernel_initializer <- kernel_initializer
        self$kernel_regularizer <- kernel_regularizer
        self$bias_initializer <- bias_initializer
        self$bias_regularizer <- bias_regularizer
      },
      
      build = function(input_shape) {

        self$glu_layers <-
          purrr::map(1L:self$num_layers, function(name) {
            layer_glu(
              filters = self$filters,
              kernel_size = self$kernel_size,
              kernel_initializer = self$kernel_initializer,
              kernel_regularizer = self$kernel_regularizer,
              bias_initializer = self$bias_initializer,
              bias_regularizer = self$bias_regularizer,
              name = paste0("glu_", name)
            )
          })
      },
      
      call = function(x, mask = NULL) {

        output <- x
        
        for (layer in self$glu_layers) output <- layer(output)
        
        residual <-
          layer_conv_1d(x, filters = self$filters, kernel_size = 1L)
        
        layer_add(c(output, residual))
        
      },
      
      compute_output_shape = function(input_shape) {
        list(NULL, input_shape[[2L]], self$filters)
      },
      
      count_params = function() {
        browser()

        params <- 
          sapply(self$layers, function(x) x$count_params()) %>% 
          sum()
        
        params
      }
    )
  )


layer_glu_block <- 
  function(object,
           filters,
           num_layers = 3,
           kernel_size = 3,
           kernel_initializer = 'glorot_normal',
           kernel_regularizer = NULL,
           bias_initializer = 'zeros',
           bias_regularizer = NULL,
           name = NULL,
           trainable = TRUE) {
    
    create_layer(
      GLUBlock,
      object,
      list(
        filters = as.integer(filters),
        num_layers = as.integer(num_layers),
        kernel_size = as.integer(kernel_size),
        kernel_initializer = tf$keras$initializers$get(kernel_initializer),
        kernel_regularizer = tf$keras$regularizers$get(kernel_regularizer),
        bias_initializer = tf$keras$initializers$get(bias_initializer),
        bias_regularizer = tf$keras$regularizers$get(bias_regularizer),
        name = name,
        trainable = trainable
      )
    )
  }

