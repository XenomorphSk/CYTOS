#' logic gate inference in structure-aware mode
#'
#' This function calls logic gates in gene regulatory networks with a priori known structure (directed edges). For this aim, it takes in steady-state gene expression profiles in (0,1) interval.
#' @param  "data"         A gene expression matrix with rows as samples and columns as genes. Gene names must be given in columns. 
#' @param  "density"      Fitted density, "norm", "laplace", "logis", "cauchy"  
#' @param  "scale"        Scale parameter of the fitted density
#' @param  "true_edges"   A two-column matrix representing the true network topology. The first column for the originating and the second column for the target nodes.  
#' @param  "number_of_em_iterations"  number of em iterations
#' @return  Outputs predicted logic gates.
#' @export

wpLogicNet_structure_aware <- function(data = expression_matrix, density = density, scale = scale, true_edges = true_edges, number_of_em_iterations = NA){

  library(VGAM)
  
  true_edges           <- as.data.frame(true_edges)
  colnames(true_edges) <- c("from", "to")
  ss                   <- nrow(data)
  number_of_genes      <- ncol(data)
  gene_names           <- colnames(data)
  
  max_par              <- number_of_genes - 1
  with_all_regulators  <- "TRUE"
  
  if(is.na(number_of_em_iterations)){
    number_of_em_iterations<- 10
  }
  
  ToComputeLogic <- function(X,ORD){
    if(is.vector(X)){
      SAMP            <- X
    }else{
      SAMP            <- X[ ,c(ORD)]
    }
    
    ncol_SAMP         <- length(c(ORD))
    
    if(length(c(ORD))>1){
      SAMP_PARTITIONED  <- matrix(rep(1,  nrow(SAMP)*(2^ncol_SAMP)), nrow(SAMP)  , 2^ncol_SAMP)
    }else{
      SAMP              <- matrix(SAMP, ncol = ncol_SAMP) 
      SAMP_PARTITIONED  <- matrix(rep(1,length(SAMP)*(2^ncol_SAMP)), length(SAMP), 2^ncol_SAMP)
    }
    
    new_logic_index   <- 1
    for(gene_id in 1:ncol_SAMP){
      SAMP_PARTITIONED[ ,new_logic_index]  <- SAMP_PARTITIONED[ ,new_logic_index]*SAMP[ ,gene_id]
    } 
    
    for(k in 1:ncol_SAMP){
      D = combn(1:ncol_SAMP,k)  # not of elements are choosen
      for(i in 1:ncol(D)){
        index_of_not    <- t(D[,i])
        new_logic_index <- new_logic_index + 1
        
        for(j in 1:ncol(index_of_not)){
          SAMP[ ,index_of_not[j]]             <- (1 - SAMP[ ,index_of_not[j]])
        }
        
        for(gene_id in 1:ncol_SAMP){
          SAMP_PARTITIONED[ ,new_logic_index] <- SAMP_PARTITIONED[ ,new_logic_index]*SAMP[ ,gene_id]
        }
        
        if(is.vector(X)){
          SAMP            <- X
        }else{
          SAMP            <- X[ ,c(ORD)]
        }
      }
    }
    return(SAMP_PARTITIONED)
  }
  
  x                       <- list()
  LOGIC_VALUES            <- array(x,c(number_of_genes,max_par))
  
  for(gene_id in 1:number_of_genes){
    impact_set_info   <- true_edges[true_edges[ ,2]==gene_names[gene_id],  ]
    if(nrow(impact_set_info)>0){
      predictive_data <- data[  ,impact_set_info$from]
      if(is.vector(predictive_data)){
        n             <- 1
      }else{
        n             <- ncol(predictive_data)
      }
      lambda          <- 2
      #######################################################################
      for(k in 1:min(n,max_par)){ # to the highest number of causal genes
        C       <- combn(1:n,k)   # choose causal gene index
        AGRE_OUT<- c()
        for(j in 1:ncol(C)){
          initial_omega                 <- matrix(c(rep(1/(2^k),(2^k)*number_of_em_iterations)), number_of_em_iterations, 2^k)
          iteration                     <- 1
          ORD                           <- t(C[,j])
          predictive_data_partitioned   <- ToComputeLogic(predictive_data,ORD)
          target_variable               <- data[,gene_id]
          target_pro                    <- matrix(c(rep(0,ss*(2^k))), ss, 2^k)
          
          for(target_index in 1:ss){
            for(z in 1:length(predictive_data_partitioned[target_index, ])){
              if(density == "norm"){
                target_pro[target_index,z] <-   dtrunc(target_variable[target_index], "norm", a = -Inf,   b = +Inf,      mean = predictive_data_partitioned[target_index,z],    sd = scale) + 0.01
              }else if(density == "laplace"){
                target_pro[target_index,z] <-   dtrunc(target_variable[target_index], "laplace", a = -Inf, b = +Inf, location = predictive_data_partitioned[target_index,z], scale = scale) + 0.01
              }else if(density == "logis"){
                target_pro[target_index,z] <-   dtrunc(target_variable[target_index], "logis", a = -Inf,   b = +Inf, location = predictive_data_partitioned[target_index,z], scale = scale) + 0.01
              }else if(density == "cauchy"){
                target_pro[target_index,z] <-   dtrunc(target_variable[target_index], "cauchy", a = -Inf,  b = +Inf, location = predictive_data_partitioned[target_index,z], scale = scale) + 0.01
              }
            }
          }
          
          #for(target_index in 1:ss){
          #  target_pro[target_index, ] <- 1/(abs(target_variable[target_index]-predictive_data_partitioned[target_index, ])) 
          #}
          
          while(iteration < number_of_em_iterations){
            initial_omega_tmp           <- initial_omega[iteration, ]
            target_pro_weighted         <- matrix(c(rep(0,ss*(2^k))), ss, 2^k)
            for(i_v in 1:(2^k)){
              target_pro_weighted[ ,i_v]<- (initial_omega_tmp[i_v])*target_pro[ ,i_v]
            }
            target_pro_weighted_row_sum <- rowSums(target_pro_weighted, na.rm = FALSE, dims = 1)
            target_pro_weighted_scaled  <- target_pro_weighted/target_pro_weighted_row_sum
            initial_omega_updated       <- colSums(target_pro_weighted_scaled, na.rm = FALSE, dims = 1)/ss
            iteration                   <- iteration +1
            initial_omega[iteration, ]  <- initial_omega_updated
          }
          target_pro_weighted           <- matrix(c(rep(0,ss*(2^k))), ss, 2^k)
          for(i_v in 1:(2^k)){
            target_pro_weighted[ ,i_v]  <- (initial_omega_updated[i_v])*target_pro[ ,i_v]
          }
          target_pro_weighted_row_sum   <- rowSums(target_pro_weighted, na.rm = FALSE, dims = 1)
          weight_rescaling_par          <- sum(target_pro_weighted_row_sum)/(ss*dlogis(0, location = 0 , scale = scale))
          
          sum_log_likelihood            <- (-1)*sum(log10(target_pro_weighted_row_sum))
          BIC                           <- 2*sum_log_likelihood + (2^k)*log10(ss)
          AIC                           <- 2*sum_log_likelihood + (2^k)*2
          #################################### random case situation #######################################
          target_pro_base                  <- matrix(c(rep(0,ss*(2^k))), ss, 2^k)
          for(target_index in 1:ss){
            if(density == "norm"){
              target_pro_base[target_index, ] <- dnorm(target_variable[target_index]   , mean     = rep(1/(2^k),(2^k)) , sd = scale) + 0.01
            }else if(density == "laplace"){
              target_pro_base[target_index, ] <- dlaplace(target_variable[target_index], location = rep(1/(2^k),(2^k)) , scale = scale) + 0.01
            }else if(density == "logis"){
              target_pro_base[target_index, ] <- dlogis(target_variable[target_index]  , location = rep(1/(2^k),(2^k)) , scale = scale) + 0.01
            }else if(density == "cauchy"){
              target_pro_base[target_index, ] <- dcauchy(target_variable[target_index] , location = rep(1/(2^k),(2^k)) , scale = scale) + 0.01
            }
          }
          initial_omega                    <- c(rep(1/(2^k),(2^k)))
          for(i_v in 1:(2^k)){
            target_pro_base[ ,i_v]         <- (initial_omega[i_v])*target_pro_base[ ,i_v]
          }
          sum_log_likelihood_base          <- (-1)*sum(log10(rowSums(target_pro_base, na.rm = FALSE, dims = 1)))
          ###################################################################################################
          AGRE_OUT                     <- rbind(AGRE_OUT, c(ORD, initial_omega_updated, sum_log_likelihood_base, sum_log_likelihood, AIC))
        }
        LOGIC_VALUES[[gene_id,k]]<-AGRE_OUT
      } # end for k
      #############################################################
    } # end if nrow >0 
  } # end for gene_id
  
  x               <- list()
  OUTPUT          <- array(x,c(number_of_genes,max_par))
  BEST_LIKELIHOODS<- matrix(c(rep(0,number_of_genes*max_par)), number_of_genes, max_par)
  BEST_BFS        <- matrix(c(rep(0,number_of_genes*max_par)), number_of_genes, max_par)
  
  for(gene_id in 1:number_of_genes){
    impact_set_info   <- true_edges[true_edges[ ,2]==gene_names[gene_id],  ]
    if(nrow(impact_set_info)>0){
      predictive_data <- data[  ,impact_set_info$from]
      if(is.vector(predictive_data)){
        n             <- 1
      }else{
        n             <- ncol(predictive_data)
      }
      
      for(i in 1:min(n,max_par)){
        RES                          <- LOGIC_VALUES[[gene_id,i]]
        
        if(nrow(RES)>1){
          I                          <- sort(RES[,ncol(RES)], decreasing = FALSE, index.return = TRUE)$ix
          SORTED_RES                 <- RES[I,]
          BEST_LIKELIHOODS[gene_id,i]<- SORTED_RES[1,ncol(SORTED_RES)-1]   # save best likelihood value
          #BEST_LIKELIHOODS[gene_id,i]<- SORTED_RES[1,ncol(SORTED_RES)]    # save best BIC value
          
          BF_COL                     <- SORTED_RES[                ,ncol(SORTED_RES)-2]-SORTED_RES[ ,ncol(SORTED_RES)-1] # calculate BF with    log u/v a = log u a - log v a
          #BF_COL                    <- SORTED_RES[nrow(SORTED_RES),ncol(SORTED_RES)-1]-SORTED_RES[ ,ncol(SORTED_RES)-1] # calculate BF with    log u/v a = log u a - log v a
          SORTED_RES                 <- cbind(SORTED_RES, BF_COL)
          BEST_BFS[gene_id,i]        <- SORTED_RES[1,ncol(SORTED_RES)]
          
          OUTPUT[[gene_id,i]]        <- SORTED_RES
        }else if(nrow(RES)==1){
          BEST_LIKELIHOODS[gene_id,i]<- RES[1, ncol(RES)-1]   # save best likelihood value
          OUTPUT[[gene_id,i]]        <- cbind(RES, RES[1, ncol(RES)-2]-RES[1, ncol(RES)-1])
          BEST_BFS[gene_id,i]        <-            RES[1, ncol(RES)-2]-RES[1, ncol(RES)-1]
        }
      } # end for i
    } # end if nrow >0 
  } # end for gene_id
  
  EM_SAMPLING <- function(data, gene_id, predicted_impact_set, number_of_em_iterations, scale, true_edges){
    impact_set_info               <- true_edges[true_edges[ ,2]==gene_names[gene_id],  ]
    if(nrow(impact_set_info)>0){
      predictive_data             <- data[  ,impact_set_info$from]
      # if(is.vector(predictive_data)){
      #   n   <- 1
      # }else{
      #   n   <- ncol(predictive_data)
      # }
      
      ss                            <- nrow(data)
      k                             <- length(predicted_impact_set)
      initial_omega                 <- matrix(c(rep(1/(2^k),(2^k)*number_of_em_iterations)), number_of_em_iterations, 2^k)
      iteration                     <- 1
      ORD                           <- predicted_impact_set
      predictive_data_partitioned   <- ToComputeLogic(predictive_data,ORD)
      target_variable               <- data[   ,gene_id]
      target_pro                    <- matrix(c(rep(0,ss*(2^k))), ss, 2^k)
      
      for(target_index in 1:ss){
        for(z in 1:length(predictive_data_partitioned[target_index, ])){
          if(density == "norm"){
            target_pro[target_index,z] <-  dtrunc(target_variable[target_index], "norm"   , a = -Inf, b = +Inf,     mean = predictive_data_partitioned[target_index,z], sd = scale) + 0.01
          }else if(density == "laplace"){
            target_pro[target_index,z] <-  dtrunc(target_variable[target_index], "laplace", a = -Inf, b = +Inf, location = predictive_data_partitioned[target_index,z], scale = scale) + 0.01
          }else if(density == "logis"){
            target_pro[target_index,z] <-  dtrunc(target_variable[target_index], "logis", a = -Inf, b = +Inf  , location = predictive_data_partitioned[target_index,z], scale = scale) + 0.01
          }else if(density == "cauchy"){
            target_pro[target_index,z] <-  dtrunc(target_variable[target_index], "cauchy", a = -Inf, b = +Inf,  location = predictive_data_partitioned[target_index,z], scale = scale) + 0.01
          }
        }
      }
      while(iteration < number_of_em_iterations){
        initial_omega_tmp           <- initial_omega[iteration, ]
        target_pro_weighted         <- matrix(c(rep(0,ss*(2^k))), ss, 2^k)
        for(i_v in 1:(2^k)){
          target_pro_weighted[ ,i_v]<- (initial_omega_tmp[i_v])*target_pro[ ,i_v]
        }
        target_pro_weighted_row_sum <- rowSums(target_pro_weighted, na.rm = FALSE, dims = 1)
        target_pro_weighted_scaled  <- target_pro_weighted/target_pro_weighted_row_sum
        initial_omega_updated       <- colSums(target_pro_weighted_scaled, na.rm = FALSE, dims = 1)/ss
        iteration                   <- iteration +1
        initial_omega[iteration, ]  <- initial_omega_updated
      }
    } # end if nrow >0
    em_res                           <- list()
    em_res$target_pro_weighted_scaled<- target_pro_weighted_scaled
    em_res$initial_omega             <- initial_omega
    return(em_res)
  }
  
  W2SYMBOL <- function(logic_significance, predicted_impact_set){
    k                  <- length(predicted_impact_set)
    reg_char           <- predicted_impact_set #as.character(paste0("G",predicted_impact_set))
    x                  <- list()
    logical_sets       <- array(x,c(2^k,2))
    
    logical_sets_index <- 1
    logical_sets[[1,1]]<- 0
    logical_sets[[1,2]]<- paste(reg_char, collapse = '.')
    
    for(i in 1:k){
      not_comb <- combn(1:k,i)  # not of elements are choosen
      for(j in 1:ncol(not_comb)){
        reg_char_tmp                        <- reg_char
        logical_sets_index                  <- logical_sets_index + 1
        logical_sets[[logical_sets_index,1]]<- not_comb[ ,j]
        
        for(l in 1:(length(not_comb[ ,j]))){
          reg_char_tmp[not_comb[ ,j][l]]    <- paste0("~", reg_char[not_comb[ ,j][l]])
        }
        
        logical_sets[[logical_sets_index,2]]<- paste0(reg_char_tmp, collapse = '.')
      }
    }
    
    LOGIC_VECTOR       <- c()
    for(i in 1:(2^k)){
      if(logic_significance[i]==1){
        LOGIC_VECTOR   <- c(LOGIC_VECTOR, logical_sets[[i,2]])
      }
    }
    return(LOGIC_VECTOR)
  }
  
  BF_optimal                     <- c() 
  likelihood_optimal             <- c() 
  x                              <- list()
  PREDICTED_IMPACT_SET           <- array(x,c(number_of_genes))
  PREDICTED_LOGIC                <- array(x,c(number_of_genes,3))
  
  predicted_gates                <- c()
  a_posteriori_distributed_sample<- list() 
  
  for(gene_id in 1:number_of_genes){
    impact_set_info              <- true_edges[true_edges[ ,2]==gene_names[gene_id],  ]
    input_edges                  <- impact_set_info$from
    predicted_impact_set         <- c()
    
    if(nrow(impact_set_info)>0){
      predictive_data            <- data[  ,impact_set_info$from]
      if(is.vector(predictive_data)){
        n   <- 1
      }else{
        n   <- ncol(predictive_data)
      }
      I                          <- sort(BEST_LIKELIHOODS[gene_id,1:min(n,max_par)] , decreasing = FALSE, index.return = TRUE)$ix
      
      if(with_all_regulators == "TRUE"){
        M                        <- OUTPUT[[gene_id,n]]
        predicted_impact_set     <- M[1,1:n]
      }else{
        M                        <- OUTPUT[[gene_id,I[1]]]
        predicted_impact_set     <- M[1,1:I[1]]
      }
      #(data, gene_id, predicted_impact_set, number_of_em_iterations, scale, true_edges)
      em_res                     <- EM_SAMPLING(data, gene_id, predicted_impact_set, number_of_em_iterations, scale, true_edges)
      omega_estimations          <- em_res$initial_omega
      target_pro_weighted_scaled <- em_res$target_pro_weighted_scaled
      
      target_pro_max_posterior   <- matrix(c(rep(0,ss*(2^length(predicted_impact_set)))), ss, 2^length(predicted_impact_set))
      for(i in 1:ss){
        I_posterior              <-  sort(target_pro_weighted_scaled[i, ] , decreasing = TRUE, index.return = TRUE)$ix
        target_pro_max_posterior[i, I_posterior[1]] <- 1
      }
      target_pro_max_posterior_colsum<- colSums(target_pro_max_posterior, na.rm = FALSE, dims = 1)
      logic_significance             <- c(rep(0,length(target_pro_max_posterior_colsum)))
      logic_significance[which(target_pro_max_posterior_colsum>0)] <- 1
      number_of_model_parameters     <- sum(logic_significance)
      distributed_sample_tmp         <- target_pro_max_posterior_colsum[target_pro_max_posterior_colsum>0]
      
      # if(M[1,ncol(M)]>critical_value){
      #   
      #   NE                          <- length(predicted_impact_set)
      #   BF_optimal                  <- c(BF_optimal, NE*M[1,ncol(M)]) 
      #   likelihood_optimal          <- c(likelihood_optimal, NE*M[1,ncol(M)-2]) 
      #   
      #   for(j in 1:length(predicted_impact_set)){
      #       predicted_impact_set[j] <- input_edges[predicted_impact_set[j]]
      #   }
      #   
      #   LOGIC_VECTOR                <- W2SYMBOL(logic_significance, predicted_impact_set, gene_names)
      # }else{
      #   predicted_impact_set        <- c()
      # }
      
      predicted_impact_set            <- input_edges[predicted_impact_set]
      LOGIC_VECTOR                    <- W2SYMBOL(logic_significance, predicted_impact_set)
      
      PREDICTED_IMPACT_SET[[gene_id]] <- predicted_impact_set
      if(length(predicted_impact_set)>0){
        PREDICTED_LOGIC[[gene_id, 1]] <- predicted_impact_set
        PREDICTED_LOGIC[[gene_id, 2]] <- logic_significance
        PREDICTED_LOGIC[[gene_id, 3]] <- LOGIC_VECTOR 
      }
      
      if(number_of_model_parameters>1){
        LOGIC_VECTOR         <- paste0(LOGIC_VECTOR, collapse = ' v ')
      }
      predicted_gates                                       <- rbind(predicted_gates, c(gene_names[gene_id], c(M[1,c(ncol(M)-3, ncol(M)-2, ncol(M))], LOGIC_VECTOR)))
      a_posteriori_distributed_sample[[gene_names[gene_id]]]<- distributed_sample_tmp
    } # end if nrow >0
  } # end for gene_id
  
  predicted_gates            <- as.data.frame(predicted_gates)
  colnames(predicted_gates)  <- c("gene_name", "-log10 M0", "-log10 M1", "log10 BF", "logic_gate")
  predicted_gates$`-log10 M0`<- round(as.numeric(predicted_gates$`-log10 M0`),digits=2)
  predicted_gates$`-log10 M1`<- round(as.numeric(predicted_gates$`-log10 M1`),digits=2)
  predicted_gates$`log10 BF` <- round(as.numeric(predicted_gates$`log10 BF`),digits=2)
  
  PREDICTED_IMPACT_SET[[gene_id+1]]   <- c(-1,-1)
  
  # PREDICTED_EDGES=c()
  # for(gene_id in 1:number_of_genes){
  #   L   <- length(PREDICTED_IMPACT_SET[[gene_id]])
  #   imp <- PREDICTED_IMPACT_SET[[gene_id]]
  #   if(L>0){
  #     for(i in 1:L){
  #       PREDICTED_EDGES=rbind(PREDICTED_EDGES, c(imp[i],gene_id))
  #     }
  #   }
  # }
  
  BF_LIKELIHOOD          <- as.data.frame(cbind(BF_optimal, likelihood_optimal))
  
  res <-list()
  # res$PREDICTED_EDGES  <- PREDICTED_EDGES
  # res$BF_LIKELIHOOD    <- BF_LIKELIHOOD
  # res$OUTPUT           <- OUTPUT
  # res$BEST_LIKELIHOODS <- BEST_LIKELIHOODS
  # res$BEST_BFS         <- BEST_BFS
  # res$PREDICTED_LOGIC  <- PREDICTED_LOGIC
  res$predicted_gates                 <- predicted_gates
  res$a_posteriori_distributed_sample <- a_posteriori_distributed_sample
  return(res)
}




