```
题目描述

输入一个链表，输出该链表中倒数第k个结点。

解题思路

双指针法，前指针比后指针先走k步，当前指针到达尾端，后指针也就到达了倒数第k个
```

```C++
class Solution {
public:
    ListNode* FindKthToTail(ListNode* pListHead, unsigned int k) {
        if(pListHead== nullptr || k == 0)
            return nullptr;
        ListNode *pre = pListHead, *p = pListHead;
        for(int i = 0; i < k; ++ i){
            if(p == nullptr)
                return nullptr;
            p = p->next;
        }
        while(p != nullptr){
            pre = pre->next;
            p = p->next;
        }
        return pre;
    }
};


```


```python
class Solution:
    def FindKthToTail(self, head, k):
        # write code here
        p, q = head, head
        for i in range(k):
            if q == None:
                return None
            q = q.next
        while q != None:
            p = p.next
            q = q.next
        return p


```
