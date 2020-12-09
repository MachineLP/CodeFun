
```
You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

Example:

Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
Explanation: 342 + 465 = 807.
```


```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        sum = 0
        carry = 0
        h = ListNode(-1)
        t = h
        while (l1!=None or l2!=None):
            if l1 != None:
                x = l1.val
                l1 = l1.next
            else:
                x = 0
            if l2 != None:
                y = l2.val
                l2 = l2.next
            else:
                y = 0
            sum = carry + x + y
            
            t.next = ListNode(sum%10)
            t = t.next
            # 进位
            carry = sum/10
        
        if carry > 0:
            t.next = ListNode(carry%10)
        return h.next
            
```
