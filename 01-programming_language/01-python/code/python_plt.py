# encoding=utf-8
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib
zhfont1 = matplotlib.font_manager.FontProperties(fname='./simsun.ttc')
names = ['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016']
 

data = [
    ['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016'],
    [40.75, 55.48, 	48.60, 	56.54,	57.36,  55.26, 	56.33, 	58.05, 	53.81, 	53.50, 	55.17 ],
    [56.74, 43.86,	44.62, 	46.03, 	44.97, 	46.08, 	43.89, 	42.00, 	45.19, 	42.33, 	46.67 ],
    [45.15, 41.32,	43.37, 	43.15, 	43.36, 	43.38, 	45.44, 	47.62, 	44.72, 	43.91, 	47.98 ],
]


# x = ['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016']
x = range(len(names))
#plt.plot(x, y, 'ro-')
#plt.plot(x, y1, 'bo-')
#pl.xlim(-1, 11)  # 限定横轴的范围
#pl.ylim(-1, 110)  # 限定纵轴的范围
 
 
plt.plot(x, data[1], marker='o', mec='r', mfc='w',label='东部')
plt.plot(x, data[2], marker='*', ms=10,label='中部')
plt.plot(x, data[3], marker='+', mec='r', mfc='w',label='西部')

plt.legend(prop=zhfont1)  # 让图例生效
plt.xticks(x, names, rotation=1, fontsize=10)
 
plt.margins(0)
plt.subplots_adjust(bottom=0.10)
plt.xlabel('年份', fontproperties=zhfont1, fontsize=10) #X轴标签
plt.ylabel("分地区指数", fontproperties=zhfont1, fontsize=10) #Y轴标签
pyplot.yticks([40, 45, 50, 55, 60])
#plt.title("A simple plot") #标题
plt.savefig('./f1.png',dpi = 900)
plt.show()

