
# Random Guessing ---------------------------------------------------------

dtt = read.table('/Users/moran/Google_Drive/Course/628/Proj2/final.txt')
counter = apply(dtt,2,sum)
plot(counter,ylim = c(0,5000))
counter = apply(dtt,1,function(x) sum(x * c(1,2,3,4,5))/1000)
mean(counter)
