
```
Given a linked list, remove the n-th node from the end of list and return its head.

Example:

Given linked list: 1->2->3->4->5, and n = 2.

After removing the second node from the end, the linked list becomes 1->2->3->5.
Note:

Given n will always be valid.
```

```python
# 不带头
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """

        if head == None:
            return None
        list_node = []
        temp_node = head
        while temp_node!=None:
            list_node.append(temp_node)
            temp_node = temp_node.next
        
        l_length = len(list_node)
        curr_node = list_node[l_length-n]
        if l_length-n > 0:
            prev_node = list_node[l_length-n-1]
            prev_node.next = curr_node.next
        else:
            head = curr_node.next
        
        return head
```


```python
# 带头
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """

        if head == None:
            return None
        list_node = []
        fake_head = ListNode(0)
        fake_head.next = head
        temp_node = fake_head
        while temp_node!=None:
            list_node.append(temp_node)
            temp_node = temp_node.next
        
        l_length = len(list_node)
        curr_node = list_node[l_length-n]
        prev_node = list_node[l_length-n-1]
        prev_node.next = curr_node.next
        
        return fake_head.next
        
```


```python
# 带头
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """

        if head == None:
            return None
        fake_head = ListNode(0)
        fake_head.next = head
        
        p1 = p2 = fake_head
        
        for i in range(n):
            if p2==None: return None
            p2 = p2.next
        while p2.next!=None:
            p2 = p2.next
            p1 = p1.next
        p1.next = p1.next.next
        
        return fake_head.next

```

