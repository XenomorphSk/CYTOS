options(pager = "cat")
.libPaths(c("~/R/library", .libPaths()))
suppressMessages(library(yeastCC, lib.loc="~/R/library"))

cat("=== Carregando o objeto principal ===\n")
data(spYCCES)
print(spYCCES)

cat("\n=== Primeiros nomes de gene (featureNames) ===\n")
print(head(featureNames(spYCCES), 10))

cat("\n=== Colunas de pData disponiveis ===\n")
print(colnames(pData(spYCCES)))

cat("\n=== Primeiras linhas de pData ===\n")
print(head(pData(spYCCES), 20))
