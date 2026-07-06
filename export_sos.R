.libPaths(c("~/R/library", .libPaths()))

# carrega os dados diretamente do arquivo sem instalar o pacote
load("wpLogicNet/data/data.rda")

cat("Objetos carregados:\n")
print(names(data))

cat("\nSOS1 - dimensoes:\n")
print(dim(data$SOS1))
cat("Primeiras linhas SOS1:\n")
print(head(data$SOS1))

cat("\nSOS2 - dimensoes:\n")
print(dim(data$SOS2))

write.csv(data$SOS1, "sos1_expression.csv")
write.csv(data$SOS2, "sos2_expression.csv")
cat("\nSalvo: sos1_expression.csv e sos2_expression.csv\n")
