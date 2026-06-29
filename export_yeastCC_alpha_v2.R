options(pager = "cat")
.libPaths(c("~/R/library", .libPaths()))
suppressMessages(library(yeastCC, lib.loc="~/R/library"))

data(spYCCES)
pd <- pData(spYCCES)
alpha_idx <- which(pd$syncmeth == "alpha")
alpha_idx <- alpha_idx[order(pd$time[alpha_idx])]

expr_alpha <- t(exprs(spYCCES)[, alpha_idx])
n_na_per_gene <- colSums(is.na(expr_alpha))
expr_complete <- expr_alpha[, n_na_per_gene == 0]

write.csv(expr_complete, "spellman_alpha_complete_genes.csv", row.names = FALSE)
cat("Salvo SEM coluna de indice. Dimensoes:", dim(expr_complete), "\n")
