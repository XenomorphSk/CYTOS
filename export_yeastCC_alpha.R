options(pager = "cat")
.libPaths(c("~/R/library", .libPaths()))
suppressMessages(library(yeastCC, lib.loc="~/R/library"))

data(spYCCES)
pd <- pData(spYCCES)

alpha_idx <- which(pd$syncmeth == "alpha")
alpha_idx <- alpha_idx[order(pd$time[alpha_idx])]

cat("N timepoints alpha encontrados:", length(alpha_idx), "\n")
cat("Tempos (min):", pd$time[alpha_idx], "\n")

expr_alpha <- t(exprs(spYCCES)[, alpha_idx])  # timepoints x genes
rownames(expr_alpha) <- pd$time[alpha_idx]

cat("\nDimensoes antes de filtrar NA:", dim(expr_alpha), "\n")

n_na_per_gene <- colSums(is.na(expr_alpha))
cat("Genes sem NENHUM valor faltante:", sum(n_na_per_gene == 0), "de", ncol(expr_alpha), "\n")
cat("Distribuicao de NAs por gene (resumo):\n")
print(summary(n_na_per_gene))

# salva a versao COMPLETA (com NA) e a versao filtrada (so genes completos)
write.csv(expr_alpha, "spellman_alpha_full_with_na.csv", row.names = TRUE)

expr_complete <- expr_alpha[, n_na_per_gene == 0]
write.csv(expr_complete, "spellman_alpha_complete_genes.csv", row.names = TRUE)
cat("\nSalvo: spellman_alpha_complete_genes.csv, dimensoes:", dim(expr_complete), "\n")
