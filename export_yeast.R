.libPaths(c("~/R/library", .libPaths()))
install.packages("spls", repos="https://cloud.r-project.org", lib="~/R/library")
library(spls, lib.loc="~/R/library")
data(yeast)

expr <- t(yeast$y)
write.csv(expr, "spellman_alpha_factor.csv", row.names = FALSE)

cat("Salvo. Dimensoes (timepoints x genes):", dim(expr), "\n")
cat("Genes:", colnames(expr)[1:5], "...\n")
