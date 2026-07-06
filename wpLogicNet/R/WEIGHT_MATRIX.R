WEIGHT_MATRIX <- function(max_par, gene_names, OUTPUT){
  number_of_genes            <- length(gene_names)
  evaluated_rg_sets          <- 0
  for(k in 1:max_par){
    evaluated_rg_sets        <- evaluated_rg_sets + ncol(combn(1:number_of_genes, k))
  }
  
  top                        <- min(20, evaluated_rg_sets) 
  x                          <- list()
  BEST_LIKELIHOODS           <- array(x,c(number_of_genes, top))
  BEST_LIKELIHOODS_k         <- array(x,c(number_of_genes))
  BEST_LIKELIHOODS_L         <- array(x,c(number_of_genes))
  criteria                   <- 3
  weight_matrix              <- matrix(0, nrow = number_of_genes - 1 , ncol = number_of_genes)
  
  for(gene_id in 1:number_of_genes){
    BEST_LIKELIHOODS_k_tmp   <- c()
    BEST_LIKELIHOODS_L_tmp   <- c()
    likelihood_list          <- c()
    top_index                <- 1
    for(k in 1:max_par){
      if(criteria==1){
        likelihood_list      <- c(likelihood_list,  OUTPUT[[gene_id,k]][ ,k+2^k+2])  # for likelihood 
      }else if(criteria==2){
        likelihood_list      <- c(likelihood_list,  OUTPUT[[gene_id,k]][ ,k+2^k+3])  # for AIC 
      }else if(criteria==3){
        likelihood_list      <- c(likelihood_list,  OUTPUT[[gene_id,k]][ ,ncol(OUTPUT[[gene_id,k]])])  # for BF
      }
    }
    if((criteria==1)||(criteria==2)){
      likelihood_list        <- sort(likelihood_list, decreasing = FALSE)
    }else{
      likelihood_list        <- sort(likelihood_list, decreasing = TRUE)
    }
    likelihood_threshold     <- likelihood_list[top]
    
    for(k in 1:max_par){
      if(criteria==1){
        OUTPUT_filtered        <- OUTPUT[[gene_id,k]][OUTPUT[[gene_id,k]][ ,k+2^k+2]<=likelihood_threshold, ]
      }else if(criteria==2){
        OUTPUT_filtered        <- OUTPUT[[gene_id,k]][OUTPUT[[gene_id,k]][ ,k+2^k+3]<=likelihood_threshold, ]
      }else if(criteria==3){
        OUTPUT_filtered        <- OUTPUT[[gene_id,k]][OUTPUT[[gene_id,k]][ ,ncol(OUTPUT[[gene_id,k]])]>=likelihood_threshold, ]
      }
      
      if(length(OUTPUT_filtered)>0){
        if(is.vector(OUTPUT_filtered)){
          BEST_LIKELIHOODS[[gene_id, top_index]]<- OUTPUT_filtered
          BEST_LIKELIHOODS_k_tmp                <- c(BEST_LIKELIHOODS_k_tmp, k) 
          BEST_LIKELIHOODS_L_tmp                <- c(BEST_LIKELIHOODS_L_tmp, OUTPUT_filtered[length(OUTPUT_filtered)-2])
          top_index                             <- top_index + 1
        }else{
          for(r_filtered in 1:nrow(OUTPUT_filtered)){
            BEST_LIKELIHOODS[[gene_id, top_index]]<- OUTPUT_filtered[r_filtered, ]
            BEST_LIKELIHOODS_k_tmp                <- c(BEST_LIKELIHOODS_k_tmp, k)
            BEST_LIKELIHOODS_L_tmp                <- c(BEST_LIKELIHOODS_L_tmp,  OUTPUT_filtered[r_filtered, (ncol(OUTPUT_filtered)-2)])
            top_index                             <- top_index + 1
          }
        }
      } # if len output filtered >0
    } # k
    BEST_LIKELIHOODS_k[[gene_id]] <- BEST_LIKELIHOODS_k_tmp
    BEST_LIKELIHOODS_L[[gene_id]] <- BEST_LIKELIHOODS_L_tmp
    ############################################################################
    if((criteria==1)||(criteria==2)){
      sort_order                  <- sort(BEST_LIKELIHOODS_L[[gene_id]], decreasing = FALSE, index.return = TRUE)$ix
    }else{
      sort_order                  <- sort(BEST_LIKELIHOODS_L[[gene_id]], decreasing = TRUE, index.return = TRUE)$ix
    }
    
    BEST_LIKELIHOODS_L_reordered  <- BEST_LIKELIHOODS_L[[gene_id]][sort_order]
    BEST_LIKELIHOODS_k_reordered  <- BEST_LIKELIHOODS_k[[gene_id]][sort_order]
    
    for(reg_gene in 1:(number_of_genes-1)){
      for(i in 1:top){
        k_tmp    <-   BEST_LIKELIHOODS_k_reordered[i]
        reg      <-   BEST_LIKELIHOODS[[gene_id, sort_order[i]]][1:k_tmp]
        
        if(k_tmp==1){
          top_ranked_lg                <- which(reg == reg_gene, arr.ind = TRUE)
        }else{
          top_ranked_lg                <- which(reg == reg_gene, arr.ind = TRUE)
        }
        
        if(length(top_ranked_lg)>0){
          bf_tmp                         <- BEST_LIKELIHOODS[[gene_id, sort_order[i]]][length(BEST_LIKELIHOODS[[gene_id, sort_order[i]]])]
          weight_matrix[reg_gene,gene_id]<- bf_tmp
          break
        }
      }
    } # end for reg_gene
    ############################################################################
  } # end for gene_id 
  
  s     <- unique(sort(as.vector(as.matrix(weight_matrix)))) 
  s     <- c(0, s[1:(length(s)-1)])
  
  edge_calc <- function(input_matrix){
    true_edges <- c()
    for(i in 1:nrow(input_matrix)){
      for(j in 1:ncol(input_matrix)){
        if(i<j){
          true_edges <- rbind(true_edges, c(c(i,j)  ,input_matrix[i,j]))  
        }else{
          true_edges <- rbind(true_edges, c(c(i+1,j),input_matrix[i,j]))  
        }
      }
    }
    return(true_edges)
  }
  
  Link            <- edge_calc(weight_matrix)
  Link_index      <- sort(Link[ ,3], decreasing = TRUE, index.return = TRUE)$ix
  Link            <- as.data.frame(Link[Link_index, ])
  colnames(Link)  <- c("from", "to", "BF_score")     
  Link[ ,1]       <- gene_names[Link[ ,1]]
  Link[ ,2]       <- gene_names[Link[ ,2]]
  ##############################################################################
  return(Link)
}