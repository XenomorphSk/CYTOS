.libPaths(c("~/R/library", .libPaths()))
if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", repos="https://cloud.r-project.org", lib="~/R/library")
}
BiocManager::install("yeastCC", lib="~/R/library", update = FALSE, ask = FALSE)
