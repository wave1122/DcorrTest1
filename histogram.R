library(ggplot2)

# setwd("E:\\Dropbox\\Teaching\\Ph.D\\Shafiullah Qureshi\\Covid-19 impact\\Data\\US\\")
setwd("E:\\Seafile\\My Library\\Copy\\SCRIPTS\\Exogeneity\\Application\\output\\GDP\\US2\\")

dta <- read.csv("results_T=152_lag_smooth=5_expn=1.500000.csv", sep = ",", header = TRUE) # import data from an csv file
df_dta <- data.frame(dta) # read data to a frame

windows(width=10, height=8)
density_plot <- ggplot(df_dta, aes(x=statistics)) + labs(x = "boostrap stats", y = "  ", colour = " ") + geom_histogram(aes(y=..density..), colour="darkorange1", fill="white") + geom_density(alpha = .2, color="darkcyan", size= 1.5)

# extract all the data on the coordinates of the kernel density function
kernel_density <- ggplot(df_dta, aes(x=statistics)) + geom_density(alpha = .2, color="darkcyan", size= 1.5)
area.data <- ggplot_build(kernel_density)$data[[1]] 
xintercept <- 0.860239

density_plot + geom_area(data = area.data[which(area.data$x >= xintercept),], aes(x=x, y=y), fill = "lightyellow") + geom_vline(aes(xintercept = xintercept), color="blue", linetype="dashed", size=1) + theme_bw() + annotate(geom = "text", x = xintercept - 0.07, y = 0.26, label="sample stat", size=5, angle=90) + annotate(geom = "text", x = xintercept + 0.9, y = 0.1, label="p-value = 0.097", size=5, angle=0)

# + stat_function(fun = function(x) dnorm(x, mean =  mean(df_dta$statistics), sd = sd(df_dta$statistics)), color = "darkred", linetype = "dashed", size = 1)