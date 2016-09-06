library(RColorBrewer)
library(colorspace)
library(sciplot)
library(psych)
library(car)
library(reshape)
library(plyr)
library(ggplot2)

#subj,stimulus,timestamp,x,y,duration,prev_sacc_amplitude,aoi_label"
df = read.table("fxtn-aois.csv",sep=",",header=TRUE)

df$subj <- factor(df$subj)
df$stimulus <- factor(df$stimulus)
df$aoi_label <- factor(df$aoi_label)

attach(df)

fit_type1 <- aov(duration ~ (stimulus * aoi_label) + Error(subj/(stimulus)))
#fit_type1 <- aov(duration ~ (stimulus * aoi_label))
summary(fit_type1)
#print(model.tables(fit_type1,"means"),digits=3)
pairwise.t.test(duration, stimulus, paired=FALSE, p.adjust="bonf")
describeBy(x=duration,group=stimulus)

pdf.options(family="NimbusSan",useDingbats=FALSE)
plotName = "./duration-CI.pdf"
pdf(plotName)
bargraph.CI(stimulus,
            duration,
            group=aoi_label,
            data=df,
            split = FALSE,
            col = "black",
            angle = c(45,45),
            density = c(0,20),
            lc = FALSE,
            uc = TRUE,
            legend = TRUE,
#           ylim = c(0,40),
            xlab = "Stimulus",
            ylab = "Fixation Duration (sec; with SE)",
            cex.lab = 1.2,
            names.arg = c("painting","puntos"),
            cex.names = 1.1,
            main = "Fixation Duration in AOIs vs. Image Type"
)
dev.off()

# feom http://www.ats.ucla.edu/stat/r/faq/subset_R.htm
# just get the columns we're interested in plotting
subdf <- subset(df,select = c(stimulus,aoi_label,duration))

plotName = "./duration.pdf"
# from http://www.r-bloggers.com/using-r-barplot-with-ggplot2/
# rearrange data leaving out the variable of interest (tt in this case)
melted <- melt(subdf, id.vars=c("stimulus","aoi_label"))
# compute means of the variable of interest, grouping by stimulus
means <- ddply(melted,c("stimulus","variable"),summarise,mean=mean(value))
# compute standard error of the mean
sem <- ddply(melted,c("stimulus","variable"),summarise,
                      mean=mean(value),
                      sem=sd(value)/sqrt(length(value)))
sem <- transform(sem, lower=mean-sem, upper=mean+sem)
pdf(plotName)
ggplot(data=means,aes(x=stimulus,y=mean)) +
  geom_bar(position=position_dodge(),stat="identity",
           colour="#303030",fill="#606060",alpha=.7) +
  geom_errorbar(data=sem,aes(ymin=lower, ymax=upper),
                width=.2, size=.3, position=position_dodge(.9)) +
  theme_bw(base_size=16) +
# ylim(0,1) +
  ylab("Fixation Duration (seconds; with SE)\n") +
  xlab("\nStimulus") +
  ggtitle("Fixation Duration per Image Type\n")
dev.off()
embedFonts(plotName, "pdfwrite", outfile = plotName,
  fontpaths =
  c("/sw/share/texmf-dist/fonts/type1/urw/helvetic",
    "/usr/share/texmf/fonts/type1/urw/helvetic",
    "/usr/local/teTeX/share/texmf-dist/fonts/type1/urw/helvetic",
    "/opt/local/share/texmf-texlive/fonts/type1/urw/helvetic",
    "/usr/share/texmf-texlive/fonts/type1/urw/helvetic",
    "/usr/local/texlive/texmf-local/fonts/type1/urw/helvetic"))

plotName = "./duration-grouped.pdf"
# from http://www.r-bloggers.com/using-r-barplot-with-ggplot2/
# rearrange data leaving out the variable of interest (tt in this case)
face <- melt(subdf[which(subdf$aoi_label == 'face'), ],
                     id.vars=c("stimulus","aoi_label"))
# compute means of the variable of interest, grouping by stimulus
face.means <- ddply(face,c("stimulus","variable"),summarise,mean=mean(value))
# compute standard error of the mean
face.sem <- ddply(face,c("stimulus","variable"),summarise,
                      mean=mean(value),
                      sem=sd(value)/sqrt(length(value)))
face.sem <- transform(face.sem, lower=mean-sem, upper=mean+sem)
face.sem$aoi <- c(rep("face",length(face.means$variable)))
# rearrange data leaving out the variable of interest (tt in this case)
hands <- melt(subdf[which(subdf$aoi_label == 'hands'), ],
                     id.vars=c("stimulus","aoi_label"))
# compute means of the variable of interest, grouping by stimulus
hands.means <- ddply(hands,c("stimulus","variable"),summarise,mean=mean(value))
# compute standard error of the mean
hands.sem <- ddply(hands,c("stimulus","variable"),summarise,
                      mean=mean(value),
                      sem=sd(value)/sqrt(length(value)))
hands.sem <- transform(hands.sem, lower=mean-sem, upper=mean+sem)
hands.sem$aoi <- c(rep("hands",length(hands.means$variable)))
sem <- rbind(face.sem,hands.sem)
pdf(plotName)
ggplot(data=sem,aes(x=aoi,y=mean,fill=stimulus)) +
  # hack to hide slashes in legend
  # first drw graph with no outline (comment out colour="...")
  geom_bar(position=position_dodge(),
           stat="identity",
#          colour="#303030",fill="#606060",
           alpha=.7) +
  # then draw graph with no legend (show_guide=FALSE)
  geom_bar(position=position_dodge(),
           stat="identity",
           colour="#707070",
           show.legend=FALSE,
           alpha=.6) +
  geom_errorbar(data=sem,aes(ymin=lower, ymax=upper),
                width=.2, size=.3, position=position_dodge(.9)) +
  scale_fill_manual(values=c("#cccccc","#969696"),
                    name="AOI",
                    labels=c("hands","face")) +
  theme_bw(base_size=16) +
# ylim(0,1) +
  ylab("Fixation Duration (seconds; with SE)\n") +
  xlab("\nStimulus") +
  theme(legend.position=c(.4,.85),
        plot.title = element_text(size=20),
        legend.text = element_text(size=14),
        legend.background = element_rect(size=.3, linetype='solid'),
        axis.text=element_text(size=16), axis.title=element_text(size=18)) +
  ggtitle("Fixation Duration in AOIs per Image Type\n")
dev.off()
embedFonts(plotName, "pdfwrite", outfile = plotName,
  fontpaths =
  c("/sw/share/texmf-dist/fonts/type1/urw/helvetic",
    "/usr/share/texmf/fonts/type1/urw/helvetic",
    "/usr/local/teTeX/share/texmf-dist/fonts/type1/urw/helvetic",
    "/opt/local/share/texmf-texlive/fonts/type1/urw/helvetic",
    "/usr/share/texmf-texlive/fonts/type1/urw/helvetic",
    "/usr/local/texlive/texmf-local/fonts/type1/urw/helvetic"))
detach(df)
