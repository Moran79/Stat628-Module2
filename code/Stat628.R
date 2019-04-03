library(tidyverse)
library(lubridate)
library(reshape)
library(mltools)
library(data.table)
library(plotly)
library(TSA)
library(tseries)

# Random Guessing ---------------------------------------------------------

dtt = read.table('/Users/moran/Google_Drive/Course/628/Proj2/final.txt')
counter = apply(dtt,2,sum)
plot(counter,ylim = c(0,5000))
counter = apply(dtt,1,function(x) sum(x * c(1,2,3,4,5))/1000)
mean(counter)


# Lexicon-Based Sentiment Analysis ----------------------------------------

dat = read.csv('/Users/moran/Google_Drive/Course/628/Proj2/Submission/Just_good_and_bad/just_gnb.csv')
dat$stars = as.factor(dat$stars)
ggplot(data = dat)+
  geom_point(aes(x = jitter(good), y = jitter(bad), color = stars), size = 3)+
  scale_color_brewer(palette="Blues")
order(dat[dat$stars ==5,4], decreasing = T)[1:5]
dat[dat$stars ==5,][410,]


# Draw Plot of Witched Spoon ----------------------------------------------

dat = read.csv('/Users/moran/PycharmProjects/leetcode/df_wic.csv')%>%
  .[,c(-1,-4)]
dat$month = month.abb[month(dat$date)]
dat = dat[,-1]
dat$stars = as.factor(dat$stars)
dat = one_hot(as.data.table(dat))

dat_plot = dat %>%
  group_by(month)%>%
  summarise(star1 = sum(stars_1),
            star2 = sum(stars_2),
            star3 = sum(stars_3),
            star4 = sum(stars_4),
            star5 = sum(stars_5))
  gather(star1:star5, key = 'class', value = 'star')



ggplot(data = dat_plot, aes(x = month)) + 
  geom_point(aes(y = star1), size = 4, color = 'green')+
  geom_line(aes(y = star1,group = 1))+
  geom_point(aes(y = star2), size = 4)+
  geom_line(aes(y = star2,group = 1))+
  geom_point(aes(y = star3), size = 4)+
  geom_line(aes(y = star3,group = 1))+
  geom_point(aes(y = star4), size = 4)+
  geom_line(aes(y = star4,group = 1))+
  geom_point(aes(y = star5), size = 4)+
  geom_line(aes(y = star5,group = 1))+
  scale_x_discrete(limits = month.abb[1:12])
  

# IG-Plot -----------------------------------------------------------------

dat = read.csv('/Users/moran/PycharmProjects/leetcode/IG_word.csv')
colnames(dat) <- c('nothing', 'not', 'dry', 'favourite', 'disappoint', '!', 'love', 'best', 'amazing', 'mediocre', 'worst', 'worth_neg', 'sum')

dat_plot1 = data.frame(Count = dat$sum)
dat_plot1$star = c('One','Two','Three','Four','Five')
dat_plot1 %>% 
  ggplot(aes(x = star, y = Count, fill = star, color = star))+
  geom_col(alpha = 0.75)+
  scale_x_discrete(limits = c('One','Two','Three','Four','Five'))+
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=20))

dat = dat / dat$sum
dat = dat[,-13]
dat = as.data.frame(apply(dat,2, function(x) x / sum(x)))
dat_tmp = melt(dat)
dat_tmp$star = c('One','Two','Three','Four','Five')
dat_tmp %>%
  ggplot(aes(x = star, y = value)) +
  geom_col(aes(color = star, fill = star), alpha = 0.75)+
  scale_x_discrete(limits = c('One','Two','Three','Four','Five'))+
  labs(y = 'Scaled Comparable Frenquency')+
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=20))+
  facet_wrap(~variable,ncol=4)+
  theme(strip.text.x = element_text(size = 30))


# Good_Bad_plot -----------------------------------------------------------


dat = read.csv('/Users/moran/Google_Drive/Course/628/Proj2/Submission/Just_good_and_bad/just_gnb.csv')
dat$stars = as.factor(dat$stars)
ggplot(data = dat)+
  geom_point(aes(x = jitter(good), y = jitter(bad), color = stars), size = 3)+
  geom_abline(slope = 1, intercept = 0, linetype="dashed")+
  coord_cartesian(xlim = c(0,20), ylim = c(0, 20))+
  geom_text(x = 15, y = 17.5, label = 'y = x', size = 10)+
  scale_color_brewer(palette="BrBG") + 
  labs(x = 'Pos_num', y = 'Neg_num') +
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=20))
order(dat[dat$stars ==5,4], decreasing = T)[1:5]
dat[dat$stars ==5,][410,]




# Distribution PLots ------------------------------------------------------


# Wicked Spoon
dat = read.csv('/Users/moran/PycharmProjects/leetcode/attributes.csv',header = F)
colnames(dat) <- c('line','legs','selection','sushi','rib','sum')

dat = dat / dat$sum
dat = dat[,-ncol(dat)]
dat = as.data.frame(apply(dat,2, function(x) x / sum(x)))
dat_tmp = melt(dat)
dat_tmp$star = c('One','Two','Three','Four','Five')
dat_tmp %>%
  ggplot(aes(x = star, y = value)) +
  geom_col(aes(color = star, fill = star), alpha = 0.75)+
  scale_x_discrete(limits = c('One','Two','Three','Four','Five'))+
  labs(y = 'Scaled Comparable Frenquency')+
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=20))+
  facet_wrap(~variable,ncol=5)+
  theme(strip.text.x = element_text(size = 30))


# Other restaurant
dat = read.csv('/Users/moran/PycharmProjects/leetcode/attributes_other.csv',header = F)
colnames(dat) <- c('line','legs','selection','sushi','rib','sum')

dat = dat / dat$sum
dat = dat[,-ncol(dat)]
dat = as.data.frame(apply(dat,2, function(x) x / sum(x)))
dat_tmp = melt(dat)
dat_tmp$star = c('One','Two','Three','Four','Five')
dat_tmp %>%
  ggplot(aes(x = star, y = value)) +
  geom_col(aes(color = star, fill = star), alpha = 0.75)+
  scale_x_discrete(limits = c('One','Two','Three','Four','Five'))+
  labs(y = 'Scaled Comparable Frenquency', x = 'star in other restaurants')+
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=20))+
  facet_wrap(~variable,ncol=5)+
  theme(strip.text.x = element_text(size = 30))




# Wilcoxon two-sample test ------------------------------------------------
dat = read.csv('/Users/moran/PycharmProjects/leetcode/attributes.csv',header = F)
colnames(dat) <- c('line','legs','selection','sushi','rib','sum')
dat_reverse = as.data.frame(apply(dat,2, function(x) dat$sum-x))

# Print out the p-value
for(i in 1:5){
  x_tmp = rep(1:5, dat[,i])
  y_tmp = rep(1:5, dat_reverse[,i])
  fit = wilcox.test(x = x_tmp, y = y_tmp, alternative = 'less')
  print(fit$p.value)
}



# Chi-Square Test ---------------------------------------------------------


dat = read.csv('/Users/moran/PycharmProjects/leetcode/attributes.csv',header = F)
colnames(dat) <- c('line','legs','selection','sushi','rib','sum')
dat_reverse = as.data.frame(apply(dat,2, function(x) dat$sum-x))

# Print out the p-value
for(i in 1:5){
  dat_use = cbind(dat[,i],dat_reverse[,i])
  fit = chisq.test(dat_use)
  print(fit$p.value)
}



# Pre-defined Season ------------------------------------------------------

df_wic = read.csv('/Users/moran/PycharmProjects/leetcode/df_wic.csv')
# df_wic$date = as_datetime(df_wic$date)
# df_wic$month = month(df_wic$date)
# df_wic$season = factor(c('Spring', 'Summer','Autumn','Winter')[lubridate::quarter(df_wic$date, fiscal_start = 3)], levels = c('Spring', 'Summer','Autumn','Winter'), ordered = TRUE)
# write.csv(df_wic,'df_wic.csv', row.names = F)

df_season = df_wic %>%
  group_by(season,stars) %>%
  summarise(n = n()) %>%
  mutate(freq = n / sum(n))

ggplot(data = df_season)+
  geom_bar(aes(x = stars, y = freq, fill = as.character(stars)), position="dodge", stat = "identity", alpha = 0.75)+
  facet_wrap(~season, nrow = 1)+
  labs(x = 'star', y = 'relative frequency')+
  scale_fill_discrete(name = 'star')+
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=20))+
  theme(strip.text.x = element_text(size = 30))
  
# Chi-square test
confusion_matrix = spread(df_season[,-4], key = season, value = n)[,-1]
chisq.test(confusion_matrix)
# ANOVA test
fit = lm(stars~season, data = df_wic)
summary(fit)
anova(fit)
# Kruskal-Wallis Test
kruskal.test(stars~season, data = df_wic)

# Seasonal Trend ----------------------------------------------------------

dat = read.csv('/Users/moran/Google_Drive/Course/628/Proj2/data/generated data/df_wic.csv') %>%
  .[,c(-1,-4)]
dat$date = as_datetime(dat$date)
dat = dat[order(dat$date),]
dat$year = year(dat$date)
dat$month = month(dat$date)
dat$day = day(dat$date)

dt = dat %>%
  dplyr::group_by(year,month) %>%
  summarise(star_mean = mean(stars))
dt$date_new =  date_new = as_date(paste(dt$year,'-',dt$month,'-01', sep = ''))

pick <- function(condition){
  function(d) d %>% filter_(condition)
}

ggplot(data = dt, aes(x = date_new, y = star_mean))+
  geom_point()+
  geom_line()+
  geom_point(data = pick(~month == 12),colour = "red", size = 4)+
  labs(x = 'Time', y = 'Average Star In Each Month')+
  theme(axis.title.x = element_text(size = 20), axis.title.y = element_text(size = 20))


fit <- ts(dt$star_mean, frequency = 12, start = c(2010,12))
fit_comp <- decompose(fit)
plot(fit_comp)
Box.test(fit_comp$random[!is.na(fit_comp$random)], type = 'Ljung-Box') # p = 0.3545

df_seasonal = data.frame(seasonal = as.numeric(fit_comp$seasonal[2:13]), date = factor(month.abb, levels = month.abb, ordered = TRUE))
ggplot(data = df_seasonal,aes(x = date, y = seasonal))+
  geom_point()+
  geom_line(group = 1)+
  geom_point(data = pick(~date == 'Dec'),colour = "red", size = 7, shape = 17)+
  theme(axis.title.x = element_text(size = 20), axis.title.y = element_text(size = 20))



# Seasonal Trend Attributes -----------------------------------------------

dat_dec = read.csv('/Users/moran/PycharmProjects/leetcode/attr_dec.csv')
dat_other = read.csv('/Users/moran/PycharmProjects/leetcode/attr_other.csv')

# Hypothesis testing and percentage
percent_dec = apply(dat_dec,2,sum) / sum(dat_dec$count)
percent_other = apply(dat_other,2,sum) / sum(dat_other$count)
# Criterion: diff / sum >= 30%  -> siginificant
aaa = abs(percent_dec - percent_other) / (percent_dec + percent_other)
# Wilcoxon two-sample test ------------------------------------------------
wilcoxon_ans = numeric(ncol(dat_dec))
for(i in 2:ncol(dat_dec)-1){
  x_tmp = rep(1:5, dat_dec[,i])
  y_tmp = rep(1:5, dat_other[,i])
  fit = wilcox.test(x = x_tmp, y = y_tmp, alternative = 'less')
  wilcoxon_ans[i] = fit$p.value
}

# Chi-Square Test ---------------------------------------------------------
chi_ans = numeric(ncol(dat_dec))
for(i in 2:ncol(dat_dec)-1){
  dat_use = cbind(dat_dec[,i],dat_other[,i])
  fit = chisq.test(dat_use,simulate.p.value = TRUE) # Sample too small, use replicates
  chi_ans[i] = fit$p.value
}

# result = as.data.frame(round(t(rbind(percent_dec,percent_other,wilcoxon_ans,chi_ans)),3))
# result[,1] = paste(as.character(result[,1]*100), '%')
# result[,2] = paste(as.character(result[,2]*100), '%')
write.csv(result, 'result1.csv')

# Draw the distribution plots given a target word
target = 'flavors'
idx = which(colnames(dat_dec) == target)
dat_tmp = data.frame('dec' = dat_dec[,idx] / sum(dat_dec[,idx]), 'other' = dat_other[,idx] / sum(dat_other[,idx])) %>%
  melt()
ggplot(data = dat_tmp) + 
  geom_bar(aes(x = rep(1:5,2),y = value, fill = as.character(rep(1:5,2))), stat = "identity", alpha = 0.75)+
  labs(x = 'star', y = 'relative frequency', title = target)+
  scale_fill_discrete(name = 'star')+
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=20))+
  theme(strip.text.x = element_text(size = 30))+
  facet_wrap(~variable)


