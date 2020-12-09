
```
Merge k sorted linked lists and return it as one sorted list. Analyze and describe its complexity.

Example:

Input:
[
  1->4->5,
  1->3->4,
  2->6
]
Output: 1->1->2->3->4->4->5->6
```


```python
# 下面这种方式采用三种不同的代码

# 第一、 采用优先队列
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        
        # 采用优先队列
        from Queue import PriorityQueue
        head = point = ListNode(0)
        q = PriorityQueue()
        for l in lists:
            if l:
                # 保存头节点的值、链表
                q.put((l.val, l))
        while not q.empty():
            val, node = q.get()
            point.next = ListNode(val)
            point = point.next
            node = node.next
            if node:
                q.put((node.val, node))
        return head.next

# 第二、采用两两链表排序的方法
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        
        amount = len(lists)
        interval = 1
        while interval < amount:
            for i in range(0, amount - interval, interval * 2):
                lists[i] = self.mergeTwoLists(lists[i], lists[i + interval])
            interval *= 2
        return lists[0] if amount > 0 else lists
    
    def mergeTwoLists(self, l1, l2):
        p = ListNode(0)
        p.next = l1
        temp = p
        
        while temp.next!=None and l2!=None:
            if temp.next.val < l2.val:
                temp = temp.next
            else:
                q = ListNode(l2.val)
                q.next = temp.next
                temp.next = q
                l2 = l2.next
                temp = temp.next
        while l2!=None:
            q = ListNode(l2.val)
            q.next = temp.next
            temp.next = q
            l2 = l2.next
            temp = temp.next
        
        return p.next

# 第三、 将链表中值取出来，排序。。。
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
    
        self.nodes = []
        head = point = ListNode(0)
        for l in lists:
            while l:
                self.nodes.append(l.val)
                l = l.next
        for x in sorted(self.nodes):
            point.next = ListNode(x)
            point = point.next
        return head.next
```
