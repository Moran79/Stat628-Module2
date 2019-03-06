library(tidyverse)
library(lubridate)
library(reshape)
library(mltools)
library(data.table)

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

