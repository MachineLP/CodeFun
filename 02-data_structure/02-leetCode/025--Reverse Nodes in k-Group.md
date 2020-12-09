
```
Given a linked list, reverse the nodes of a linked list k at a time and return its modified list.

k is a positive integer and is less than or equal to the length of the linked list. If the number of nodes is not a multiple of k then left-out nodes in the end should remain as it is.

Example:

Given this linked list: 1->2->3->4->5

For k = 2, you should return: 2->1->4->3->5

For k = 3, you should return: 3->2->1->4->5

Note:

Only constant extra memory is allowed.
You may not alter the values in the list's nodes, only nodes itself may be changed.
```

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def reverseKGroup(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        if k<0: return head
        fake = ListNode(0)
        fake.next = head
        p = fake
        
        while (p):
            p.next = self.reverseList(p.next, k)
            for i in range(k):
                if p:
                    p = p.next
        return fake.next
    
    def reverseList(self, head, k):
        
        p_end = head
        while(p_end and k>0):
            p_end = p_end.next
            k -= 1
        if k>0: return head
        
        p_head = p_end
        p = head
        
        while (p!=p_end):
            q = p.next
            p.next = p_head
            p_head = p
            p = q
        return p_head
        
```
