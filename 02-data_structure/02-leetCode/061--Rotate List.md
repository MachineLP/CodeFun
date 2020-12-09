```
Given a linked list, rotate the list to the right by k places, where k is non-negative.

Example 1:

Input: 1->2->3->4->5->NULL, k = 2
Output: 4->5->1->2->3->NULL
Explanation:
rotate 1 steps to the right: 5->1->2->3->4->NULL
rotate 2 steps to the right: 4->5->1->2->3->NULL
Example 2:

Input: 0->1->2->NULL, k = 4
Output: 2->0->1->NULL
Explanation:
rotate 1 steps to the right: 2->0->1->NULL
rotate 2 steps to the right: 1->2->0->NULL
rotate 3 steps to the right: 0->1->2->NULL
rotate 4 steps to the right: 2->0->1->NULL
```

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    
    def get_listnode_length(self, head):
        length = 0
        while head:
            length += 1
            head = head.next
        return length
    
    def rotateRight(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        if not head:
            return head

        length = self.get_listnode_length(head)
        if length == 1:
            return head

        pos = length - k % length
        if pos == length:
            return head

        traversal = head
        for index in range(1, pos):
            traversal = traversal.next

        rolate = traversal.next
        traversal.next = None
        temp = rolate
        while temp.next:
            temp = temp.next
        temp.next = head

        return rolate
        
```
