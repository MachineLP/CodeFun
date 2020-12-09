# coding = utf-8

'''
    紧跟大神的学习脚步
'''
from queue import Queue


'''
    ##函数说明##
    @函数名称：base_converter
    @功能说明：将二进制转换为其他进制
    @输入参数：num: 数字，base: 基数
    @返回说明：数字的字符串
    @备注说明：
'''
def base_converter(num, base):
    digs = '0123456789ABCDEF'  # 支持16位
    base_stack = list()

    while num > 0:
        rem = num % base
        base_stack.append(rem)
        num //= base  # 递减一直到0

    res_str = ''  # 转换为str
    while base_stack:
        res_str += digs[base_stack.pop()]

    return res_str


'''
    ##函数说明##
    @函数名称：hot_potato
    @功能说明：循环去除
    @输入参数：name_list: 名字列表，num: 循环数
    @返回说明：最后剩下的人
    @备注说明：
'''
def hot_potato(name_list, num):
    q = Queue()

    for name in name_list:
        q.put(name)

    while q.qsize() > 1:
        for i in range(num - 1):  # 每次都死一个循环，最后一个死亡
            live = q.get()
            q.put(live)
        dead = q.get()  # 输出死亡
        print('Dead: {}'.format(dead))

    return q.get()

def test_of_hot_potato():
    name_list = ['Bill', 'David', 'Susan', 'Jane', 'Kent', 'Brad']
    num = 3
    print(hot_potato(name_list, num))


'''
    ##函数说明##
    @函数名称：reverse_list
    @功能说明：链表逆序，交换链表
    @输入参数：node_head: 链表头
    @返回说明：新的链表的头结点
    @备注说明：
'''
class Node:
    def __init__(self, data=None):
        self.data = data
        self.next_node = None

def reverse_list(node_head):

    prev_node = None
    # 具体的可以查看：https://blog.csdn.net/feliciafay/article/details/6841115
    while node_head:
        next_node = node_head.next_node
        node_head.next_node = prev_node
        prev_node = node_head    # prev_node往前移
        node_head = next_node    # node_head往前移

    return prev_node

def reverse_list_cp(node_head):
    cur, prev = node_head, None
    while cur:
        cur.next, prev, cur = prev, cur, cur.next
    return prev

def init_list():
    n1 = Node(1)
    n2 = Node(2)
    n3 = Node(3)
    n4 = Node(4)
    n5 = Node(5)
    n1.next_node = n2
    n2.next_node = n3
    n3.next_node = n4
    n4.next_node = n5
    return n1


def show_list(node_head):
    head = node_head
    while head:
        print(head.data, end=' ')
        head = head.next_node

def test_of_reverse_list():
    head_node = init_list()
    show_list(head_node)
    print()
    head_node = reverse_list(head_node)
    show_list(head_node)


from collections import deque
'''
    ##函数说明##
    @函数名称：reverse_list
    @功能说明：回文，双端队列
    @输入参数：a_str: 输入字符串
    @返回说明：是否回文
    @备注说明：
'''
def pal_checker(a_str):
    q_char = deque()

    for ch in a_str:
        q_char.append(ch)

    equal = True

    # while的终止条件长度或者Bool
    while len(q_char) > 1:
        first = q_char.pop()
        last = q_char.popleft()
        if first != last:
            equal = False
            break

    return equal

def test_of_pal_checker():
    print(pal_checker('lsdkjfskf'))
    print(pal_checker('radar'))


'''
    ##函数说明##
    @函数名称：par_checker
    @功能说明：括号匹配，list包含栈的功能,append是添加，pop是删除, https://docs.python.org/2/tutorial/datastructures.html  14行
    @输入参数：symbol_str: 符号字符串
    @返回说明：是否
    @备注说明：
'''
def par_checker(symbol_str):

    s = list()  # python的list可以实现stack功能
    idx = 0
    while idx < len(symbol_str):
        symbol = symbol_str[idx]
        if symbol == '(':
            s.append(symbol)
        elif symbol == ')':
            s.pop()
        idx += 1
    if not s:
        return True
    else:
        return False

def test_of_par_checker():
    print(par_checker('(())'))
    print(par_checker('((()'))
    print(par_checker('(a)()((()))'))


if __name__ == '__main__':
    test = base_converter(2, 2)
    print ('test', test)
    test_of_hot_potato()
    test_of_reverse_list()
    test_of_pal_checker()
    test_of_par_checker()

