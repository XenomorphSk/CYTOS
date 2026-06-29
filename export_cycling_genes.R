options(pager = "cat")
.libPaths(c("~/R/library", .libPaths()))
suppressMessages(library(yeastCC, lib.loc="~/R/library"))

data(spYCCES)

cat("=== Colunas de featureData (pode ter anotacao de ciclo-regulado) ===\n")
print(colnames(fData(spYCCES)))
print(head(fData(spYCCES), 5))

# Se nao tiver featureData util, exporta a lista de genes que o Spellman
# classificou como ciclo-regulados (os 800 que aparecem no paper)
# O yeastCC tambem tem um objeto chamado 'ycYCCES' (Yeung et al.) ou
# simplesmente o spYCCES pode ter um subset - vamos checar o que existe
cat("\n=== Todos os objetos disponiveis no pacote ===\n")
print(ls("package:yeastCC"))
