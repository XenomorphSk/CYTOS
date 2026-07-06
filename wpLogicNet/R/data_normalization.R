#' transforms data into (0,1) interval
#'
#' This function transforms gene expression data into (0,1) interval
#' @param  "data"         A gene expression matrix with rows as samples and columns as genes.
#' @return  normalized gene expression data
#' @export  


data_normalization  <- function(data){
  normalized_data   <- data
  for(i in 1:nrow(data)){
    for(j in 1:ncol(data)){
      normalized_data[i,j] <- min(0.999999, (data[i,j]-min(data[i, ]) + 0.01)/(max(data[i, ])-min(data[i, ]) + 0.01))
    }
  }
  return(normalized_data)
}