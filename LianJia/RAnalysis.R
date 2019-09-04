library(ggplot2)
library(Seurat)
d = read.csv('D:/Codes/PythonSkills/LianJia/contents.csv', sep = '\t')
d = d[d$price < 10000,]
pr = d$price
# 0 1 2 3 4 5 6     7     8    9      10  11  12         13      14   15      16      17
# E S W N y x space floor time subway new key decoration heating rent twobath deposit price

p1 = ggplot(d, aes(x = E, y = price)) + geom_point() + geom_smooth(method = 'lm')
p2 = ggplot(d, aes(x = S, y = price)) + geom_point() + geom_smooth(method = 'lm')
p3 = ggplot(d, aes(x = W, y = price)) + geom_point() + geom_smooth(method = 'lm')
p4 = ggplot(d, aes(x = N, y = price)) + geom_point() + geom_smooth(method = 'lm')
CombinePlots(list(p1, p2, p3, p4), ncol = 2)

tt = function(x){t.test(pr[d[,x] == 1], pr[d[,x] == 0])}
sapply(c('E', 'S', 'W', 'N'), function(name) {t.test(d$price[d[name] == 1], d$price[d[name] == 0])$p.value})

p1 = ggplot(d, aes(x = x, y = y, color = price)) + geom_point()
p1

p1 = ggplot(d, aes(x = space, y = price)) + geom_point() + geom_smooth(method = 'lm')
p1

p1 = ggplot(d, aes(x = floor, y = price)) + geom_point() + geom_smooth(method = 'lm')
p1

p1 = ggplot(d, aes(x = log(time), y = price)) + geom_point() + geom_smooth(method = 'lm')
p1

p1 = ggplot(d, aes(x = subway, y = price)) + geom_point() + geom_smooth(method = 'lm')
p2 = ggplot(d, aes(x = new, y = price)) + geom_point() + geom_smooth(method = 'lm')
p3 = ggplot(d, aes(x = key, y = price)) + geom_point() + geom_smooth(method = 'lm')
p4 = ggplot(d, aes(x = decoration, y = price)) + geom_point() + geom_smooth(method = 'lm')
p5 = ggplot(d, aes(x = heating, y = price)) + geom_point() + geom_smooth(method = 'lm')
p6 = ggplot(d, aes(x = rent, y = price)) + geom_point() + geom_smooth(method = 'lm')
p7 = ggplot(d, aes(x = twobath, y = price)) + geom_point() + geom_smooth(method = 'lm')
p8 = ggplot(d, aes(x = deposit, y = price)) + geom_point() + geom_smooth(method = 'lm')
CombinePlots(list(p1, p2, p3, p4, p5, p6, p7, p8), ncol = 2)
tt('subway')
tt('new') #
tt('key') #
tt('decoration')
tt('heating') #
tt('rent') #
tt('twobath') #
tt('deposit') #
