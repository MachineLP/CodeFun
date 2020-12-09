# 抢票模拟，使用DECR原子操作
import redis
import threading
# 创建连接池
pool = redis.ConnectionPool(host = '127.0.0.1', port=6379, db=0)
# 初始化 redis
r = redis.StrictRedis(connection_pool = pool)

# 设置KEY
KEY="ticket_count"
# 模拟第i个用户进行抢购
def sell(i):
    # 使用decr对KEY减1
    temp = r.decr(KEY)
    if temp >= 0:
        print('用户 {} 抢票成功，当前票数 {}'.format(i, temp))
    else:
        print('用户 {} 抢票失败，票卖完了'.format(i))

if __name__ == "__main__":
    # 初始化5张票
    r.set(KEY, 5)  
    # 设置8个人抢票
    for i in range(8):
        t = threading.Thread(target=sell, args=(i,))
        t.start()
