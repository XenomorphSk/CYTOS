#' structure and logic gate inference
#'
#' This function calls structure together with logic gates in gene regulatory networks with un-known structure. For this aim, it takes in steady-state gene expression profiles in (0,1) interval.
#' @param  "data"         A gene expression matrix with rows as samples and columns as genes. Gene names must be given in the columns. 
#' @param  "density"      Fitted density, "norm", "laplace", "logis", "cauchy"  
#' @param  "scale"        Scale parameter of the fitted density
#' @param  "number_of_em_iterations"  number of em iterations
#' @param  "BF_threshold" Bayes Factor threshold
#' @param  "max_num_regulators" Max number of genes in a logic gate to regulate the target gene profile
#' @return  the predicted gene-gene interactions (directed edges) and logic gates.
#' "predicted_edges_top_logics" and "predicted_gates_top_logic" are prediction in top-logics mode.
#' "a_posteriori_distributed_sample" outputs the number of samples emitted from each active partition in top-logics mode (inferred a posteriori). 
#' "top_edges" are predicted edges in top_edges mode, with column 1 as originating gene, column 2 as target gene and column 3 as the BF confidence score. 
#'  
#' @export  


wpLogicNet <- function(data = expression_matrix, density = density, scale = scale, number_of_em_iterations = NA, BF_threshold = NA, max_num_regulators = NA){
  
  library(VGAM)
  ss                   <- nrow(data)
  number_of_genes      <- ncol(data)
  gene_names           <- colnames(data)
  
  if(is.na(number_of_em_iterations)){
    number_of_em_iterations<- 10
  }
  if(is.na(BF_threshold)){
    BF_threshold       <- 2
  }
  if(is.na(max_num_regulators)){
    max_num_regulators <- number_of_genes - 2
  }

  ToComputeLogic  <- function(X,ORD){
    SAMP               <- X[ ,c(ORD)]
    ncol_SAMP          <- length(c(ORD))
    
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
        
        SAMP=X[ ,c(ORD)]
      }
    }
    return(SAMP_PARTITIONED)
  }
  
  x                       <- list()
  LOGIC_VALUES            <- array(x,c(number_of_genes,max_num_regulators))
  LOGIC_VALUES_PRO        <- array(x,c(number_of_genes,max_num_regulators))
  
  for(gene_id in 1:number_of_genes){
    predictive_data <- data[  ,-gene_id]
    m               <- nrow(predictive_data)# m sample size and n number of genes which may
    n               <- ncol(predictive_data)
    #######################################################################
    for(k in 1:max_num_regulators){         # to the highest number of causal genes
      C             <- combn(1:n,k)         # choose causal gene index
      AGRE_OUT      <- c()
      AGRE_OUT_PRO  <- c()
      for(j in 1:ncol(C)){
        initial_omega                 <- matrix(c(rep(1/(2^k),(2^k)*number_of_em_iterations)), number_of_em_iterations, 2^k)
        iteration                     <- 1
        ORD                           <- t(C[,j])
        predictive_data_partitioned   <- ToComputeLogic(predictive_data,ORD)
        target_variable               <- data[ ,gene_id]
        target_pro                    <- matrix(c(rep(0,ss*(2^k))), ss, 2^k)
        
        for(target_index in 1:ss){
          for(z in 1:length(predictive_data_partitioned[target_index, ])){
            if(density == "norm"){
              target_pro[target_index,z] <-   dtrunc(target_variable[target_index], "norm"   , a = -Inf, b = +Inf, mean = predictive_data_partitioned[target_index,z], sd = scale) + 0.01
            }else if(density == "laplace"){
              target_pro[target_index,z] <-   dtrunc(target_variable[target_index], "laplace", a = -Inf, b = +Inf, location = predictive_data_partitioned[target_index,z], scale = scale) + 0.01
            }else if(density == "logis"){
              target_pro[target_index,z] <-   dtrunc(target_variable[target_index], "logis"  , a = -Inf, b = +Inf, location = predictive_data_partitioned[target_index,z], scale = scale) + 0.01
            }else if(density == "cauchy"){
              target_pro[target_index,z] <-   dtrunc(target_variable[target_index], "cauchy" , a = -Inf, b = +Inf, location = predictive_data_partitioned[target_index,z], scale = scale) + 0.01
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
        sum_log_likelihood            <- (-1)*sum(log10(target_pro_weighted_row_sum))
        BIC                           <- 2*sum_log_likelihood + (2^k)*log10(ss)
        AIC                           <- 2*sum_log_likelihood + (2^k)*2
        #################################### random case situation #######################################
        target_pro_base                  <- matrix(c(rep(0,ss*(2^k))), ss, 2^k)
        for(target_index in 1:ss){
          if(density == "norm"){
            target_pro_base[target_index, ] <- dnorm(target_variable[target_index]   ,  mean    = rep(1/(2^k),(2^k)) , sd = scale) + 0.01
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
        AGRE_OUT_PRO                 <- rbind(AGRE_OUT_PRO, c(target_pro_weighted_row_sum))
      }
      LOGIC_VALUES[[gene_id,k]]      <- AGRE_OUT
      LOGIC_VALUES_PRO[[gene_id,k]]  <- AGRE_OUT_PRO
    }
    #############################################################
  }
  
  x               <- list()
  OUTPUT          <- array(x,c(number_of_genes,max_num_regulators))
  OUTPUT_PRO      <- array(x,c(number_of_genes,max_num_regulators))
  BEST_LIKELIHOODS<- matrix(c(rep(0,number_of_genes*max_num_regulators)), number_of_genes, max_num_regulators)
  BEST_BFS        <- matrix(c(rep(0,number_of_genes*max_num_regulators)), number_of_genes, max_num_regulators)
  
  for(gene_id in 1:number_of_genes){
    for(i in 1:max_num_regulators){
      RES                        <- LOGIC_VALUES[[gene_id,i]]
      I                          <- sort(RES[,ncol(RES)], decreasing = FALSE, index.return = TRUE)$ix
      SORTED_RES                 <- RES[I,]
      BEST_LIKELIHOODS[gene_id,i]<- SORTED_RES[1,ncol(SORTED_RES)-1]   # save best likelihood value
      #BEST_LIKELIHOODS[gene_id,i]<- SORTED_RES[1,ncol(SORTED_RES)]    # save best BIC value
      
      BF_COL                     <- SORTED_RES[                ,ncol(SORTED_RES)-2]-SORTED_RES[ ,ncol(SORTED_RES)-1] # calculate BF with    log u/v a = log u a - log v a
      #BF_COL                    <- SORTED_RES[nrow(SORTED_RES),ncol(SORTED_RES)-1]-SORTED_RES[ ,ncol(SORTED_RES)-1] # calculate BF with    log u/v a = log u a - log v a
      SORTED_RES                 <- cbind(SORTED_RES, BF_COL)
      BEST_BFS[gene_id,i]        <- SORTED_RES[1,ncol(SORTED_RES)]
      
      OUTPUT[[gene_id,i]]        <- SORTED_RES
      OUTPUT_PRO[[gene_id,i]]    <- cbind(LOGIC_VALUES_PRO[[gene_id,i]][I, ],I)
    }
  }
  
  
  EM_SAMPLING <- function(data, gene_id, predicted_impact_set, number_of_em_iterations, scale){
     predictive_data               <- data[   ,-gene_id]
     ss                            <- nrow(predictive_data)
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
           target_pro[target_index,z] <-  dtrunc(target_variable[target_index], "norm"   , a = -Inf, b = +Inf, mean = predictive_data_partitioned[target_index,z], sd = scale) + 0.01
         }else if(density == "laplace"){
           target_pro[target_index,z] <-  dtrunc(target_variable[target_index], "laplace", a = -Inf, b = +Inf, location = predictive_data_partitioned[target_index,z], scale = scale) + 0.01
         }else if(density == "logis"){
           target_pro[target_index,z] <-  dtrunc(target_variable[target_index], "logis", a = -Inf, b = +Inf  , location = predictive_data_partitioned[target_index,z], scale = scale) + 0.01
         }else if(density == "cauchy"){
           target_pro[target_index,z] <-  dtrunc(target_variable[target_index], "cauchy", a = -Inf, b = +Inf , location = predictive_data_partitioned[target_index,z], scale = scale) + 0.01
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

  BF_optimal                 <- c() 
  likelihood_optimal         <- c() 
  x                          <- list()
  PREDICTED_IMPACT_SET       <- array(x,c(number_of_genes))
  PREDICTED_LOGIC            <- array(x,c(number_of_genes,3))
  
  predicted_gates                <- c()
  a_posteriori_distributed_sample<- list() 
  
  for(gene_id in 1:number_of_genes){
    I                        <- sort(BEST_LIKELIHOODS[gene_id,] , decreasing = FALSE, index.return = TRUE)$ix
    #I                       <- sort(BEST_LIKELIHOODS[gene_id,1:min(max_num_regulators, max_num_regulator)] , decreasing = FALSE, index.return = TRUE)$ix
    
    M                        <- OUTPUT[[gene_id,I[1]]]
    predicted_impact_set     <- M[1,1:I[1]]
                                          # (data, gene_id, predicted_impact_set, number_of_em_iterations, scale)
    em_res                    <- EM_SAMPLING(data, gene_id, predicted_impact_set, number_of_em_iterations, scale)
    omega_estimations         <- em_res$initial_omega
    target_pro_weighted_scaled<- em_res$target_pro_weighted_scaled
    
    target_pro_max_posterior  <- matrix(c(rep(0,ss*(2^length(predicted_impact_set)))), ss, 2^length(predicted_impact_set))
    for(i in 1:ss){
      I_posterior             <-  sort(target_pro_weighted_scaled[i, ] , decreasing = TRUE, index.return = TRUE)$ix
      target_pro_max_posterior[i, I_posterior[1]] <- 1
    }
    target_pro_max_posterior_colsum     <- colSums(target_pro_max_posterior, na.rm = FALSE, dims = 1)
    logic_significance                  <- c(rep(0,length(target_pro_max_posterior_colsum)))
    logic_significance[which(target_pro_max_posterior_colsum>0)]<- 1
    number_of_model_parameters          <- sum(logic_significance)
    distributed_sample_tmp              <- target_pro_max_posterior_colsum[target_pro_max_posterior_colsum>0]
    
    if(M[1,ncol(M)]>BF_threshold){
      
      NE                        <- length(predicted_impact_set)
      BF_optimal                <- c(BF_optimal, NE*M[1,ncol(M)]) 
      likelihood_optimal        <- c(likelihood_optimal, NE*M[1,ncol(M)-2]) 
      
      for(j in 1:length(predicted_impact_set)){
        if(predicted_impact_set[j]>=gene_id){
           predicted_impact_set[j]=predicted_impact_set[j]+1
        }
      }
      
      predicted_impact_set           <- gene_names[predicted_impact_set]
      PREDICTED_IMPACT_SET[[gene_id]]<- predicted_impact_set
      LOGIC_VECTOR                   <- W2SYMBOL(logic_significance, predicted_impact_set)
      
      if(length(predicted_impact_set)>0){
        PREDICTED_LOGIC[[gene_id, 1]]  <- predicted_impact_set
        PREDICTED_LOGIC[[gene_id, 2]]  <- logic_significance
        PREDICTED_LOGIC[[gene_id, 3]]  <- LOGIC_VECTOR 
      }
      
      if(number_of_model_parameters>1){
        LOGIC_VECTOR         <- paste0(LOGIC_VECTOR, collapse = ' v ')
      }
      predicted_gates                                       <- rbind(predicted_gates, c(gene_names[gene_id], c(M[1,c(ncol(M)-3, ncol(M)-2, ncol(M))], LOGIC_VECTOR)))
      a_posteriori_distributed_sample[[gene_names[gene_id]]]<- distributed_sample_tmp
      
    }else{
      predicted_impact_set   <- c()
    }
  } # end for gene_id
  
  predicted_gates            <- as.data.frame(predicted_gates)
  colnames(predicted_gates)  <- c("gene_name", "-log10 M0", "-log10 M1", "log10 BF", "logic_gate")
  predicted_gates$`-log10 M0`<- round(as.numeric(predicted_gates$`-log10 M0`),digits=2)
  predicted_gates$`-log10 M1`<- round(as.numeric(predicted_gates$`-log10 M1`),digits=2)
  predicted_gates$`log10 BF` <- round(as.numeric(predicted_gates$`log10 BF`),digits=2)
  
  PREDICTED_IMPACT_SET[[gene_id+1]]   <- c(-1,-1)
  
  predicted_edges <- c()
  for(gene_id in 1:number_of_genes){
    L     <- length(PREDICTED_IMPACT_SET[[gene_id]])
    imp   <- PREDICTED_IMPACT_SET[[gene_id]]
    if(L>0){
      for(i in 1:L){
        predicted_edges <- rbind(predicted_edges, c(imp[i],gene_names[gene_id]))
      }
    }
  }
  
  
  BF_LIKELIHOOD          <- as.data.frame(cbind(BF_optimal, likelihood_optimal))
  link                   <- WEIGHT_MATRIX(max_num_regulators, gene_names, OUTPUT)
  
  
  res <-list()
  
  # res$OUTPUT           <- OUTPUT
  # res$PREDICTED_LOGIC  <- PREDICTED_LOGIC
  # res$BF_LIKELIHOOD    <- BF_LIKELIHOOD
  # res$BEST_LIKELIHOODS <- BEST_LIKELIHOODS
  # res$BEST_BFS         <- BEST_BFS
  # res$OUTPUT_PRO       <- OUTPUT_PRO
  # res$LOGIC_VALUES_PRO <- LOGIC_VALUES_PRO
  
  res$predicted_edges_top_logics    <- predicted_edges
  res$predicted_gates_top_logics    <- predicted_gates
  res$a_posteriori_distributed_sample <- a_posteriori_distributed_sample
  res$top_edges                     <- link
  
  return(res)
}