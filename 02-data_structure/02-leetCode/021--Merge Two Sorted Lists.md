
```
Merge two sorted linked lists and return it as a new list. The new list should be made by splicing together the nodes of the first two lists.

Example:

Input: 1->2->4, 1->3->4
Output: 1->1->2->3->4->4
```

```python
# 链表的题目套路还是很多的， 带头节点和不带头节点会有差别，为了方便可以将不带头节点的链表加入头节点。
# 该题的方案可以考虑使用带头节点的链表会更简单
# 另外还要考虑额外申请一个链表，还是在原有链表上修改。

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
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
```
