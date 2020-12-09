# coding:utf-8
# 下载豆瓣 王祖贤的海报
import requests
import json

query = '王祖贤'
''' 下载图片 '''
def download(src, id):
	dir = './' + str(id) + '.jpg'
	try:
		pic = requests.get(src, timeout=10)
	except requests.exceptions.ConnectionError:
		print('图片无法下载')

	fp = open(dir, 'wb')
	fp.write(pic.content)
	fp.close()

''' for循环 请求全部的url '''
for i in range(0, 22471, 20):
	url = 'https://www.douban.com/j/search_photo?q='+query+'&limit=20&start='+str(i)
	html = requests.get(url).text    #得到返回结果
	response = json.loads(html,encoding='utf-8') #将JSON格式转换成Python对象
	for image in response['images']:
		print(image['src']) #查看当前下载的图片网址
		download(image['src'], image['id']) #下载一张图片
		
