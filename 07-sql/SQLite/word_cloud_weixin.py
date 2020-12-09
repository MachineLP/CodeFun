import sqlite3
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import jieba
import os
import re

# 去掉停用词
def remove_stop_words(f):
	stop_words = ['你好', '已添加', '现在', '可以', '开始', '聊天', '当前', '群聊', '人数', '过多', '显示', '群成员', '昵称', '信息页', '关闭', '参与人', '还有', '嗯']
	for stop_word in stop_words:
		f = f.replace(stop_word, '')
	return f

# 生成词云
def create_word_cloud(f):
	print('根据微信聊天记录，生成词云!')
	# 设置本地的simhei字体文件位置
	FONT_PATH = os.environ.get("FONT_PATH", os.path.join(os.path.dirname(__file__), "simhei.ttf"))
	f = remove_stop_words(f)
	cut_text = " ".join(jieba.cut(f,cut_all=False, HMM=True))
	wc = WordCloud(
		font_path=FONT_PATH,
		max_words=100,
		width=2000,
		height=1200,
    )
	wordcloud = wc.generate(cut_text)
	# 写词云图片
	wordcloud.to_file("wordcloud.jpg")
	# 显示词云文件
	plt.imshow(wordcloud)
	plt.axis("off")
	plt.show()

def get_content_from_weixin():
	# 创建数据库连接
	# 这里需要把 找到的weixin.db 放到根目录，具体方法之前文稿讲过
	conn = sqlite3.connect("weixin.db")
	# 获取游标
	cur = conn.cursor()
	# 创建数据表
	# 查询当前数据库中的所有数据表
	sql = "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'Chat\_%' escape '\\\'"
	cur.execute(sql)
	tables = cur.fetchall()
	content = ''
	for table in tables:
	    sql = "SELECT Message FROM " + table[0]
	    print(sql)
	    cur.execute(sql)
	    temp_result = cur.fetchall()
	    for temp in temp_result:
	    	content = content + str(temp)
	# 提交事务 
	conn.commit()
	# 关闭游标
	cur.close()
	# 关闭数据库连接
	conn.close()
	return content
content = get_content_from_weixin()
# 去掉HTML标签里的内容
pattern = re.compile(r'<[^>]+>',re.S)
content = pattern.sub('', content)
# 将聊天记录生成词云
create_word_cloud(content)