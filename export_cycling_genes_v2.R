options(pager = "cat")
.libPaths(c("~/R/library", .libPaths()))
suppressMessages(library(yeastCC, lib.loc="~/R/library"))

data(orf800)
cat("Tipo do objeto orf800:", class(orf800), "\n")
cat("Tamanho:", length(orf800), "\n")
cat("Primeiros 10:", head(orf800, 10), "\n")
cat("Ultimos 5:", tail(orf800, 5), "\n")

# Salva a lista pra Python usar depois
write.csv(data.frame(orf=orf800), "orf800_cycling_genes.csv", row.names=FALSE)
cat("\nSalvo em orf800_cycling_genes.csv\n")
