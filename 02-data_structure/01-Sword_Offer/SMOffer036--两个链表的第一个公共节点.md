```
题目描述

输入两个链表，找出它们的第一个公共结点。

解题思路

首先遍历两个链表得他们的长度，再设两个指针分别指向他们的头结点，将较长的链表的头指针先遍历两个长度之差个节点，再同时遍历两个链表，找到的第一个相同的节点，即为他们的第一个公共的节点。
```

```C++
class Solution {
public:
    ListNode* FindFirstCommonNode( ListNode* pHead1, ListNode* pHead2) {
        int len1 = 0, len2 = 0;
        ListNode *p = pHead1, *q = pHead2;
        while(p != nullptr){
            ++ len1;
            p = p->next;
        }
        while(q != nullptr){
            ++ len2;
            q = q->next;
        }
        if(len1 < len2)
            p = pHead1, q = pHead2;
        else
            p = pHead2, q = pHead1;
        for(int i = 0; i < abs(len2 - len1); ++ i)
            q = q->next;
        while(p != q){
            p = p->next;
            q = q->next;
        }
        return p;
    }
};


```

```python
class Solution:
    def FindFirstCommonNode(self, pHead1, pHead2):
        # write code here
        if pHead1 == None or pHead2 == None:
            return None
        p = pHead1
        q = pHead2
        len1, len2 = 0, 0
        while p != None:
            len1 += 1
            p = p.next
        while q != None:
            len2 += 1
            q = q.next
        p, q = (pHead1,pHead2) if len1 > len2 else (pHead2, pHead1)
        for i in range(abs(len1 - len2)):
            p = p.next
        while p != q:
            p = p.next
            q = q.next
        return p


```
