```
题目描述

输入一个链表，反转链表后，输出新链表的表头。

解题思路

头插法重新排列链表，我们需要考虑到几种特殊情况：①pHead为空指针；②输入链表只有一个节点
```

```C++
class Solution {
public:
    ListNode* ReverseList(ListNode* pHead) {
        ListNode *pReversedHead = nullptr, *pNode = pHead, *pPrev = nullptr;
        while(pNode != nullptr){
            ListNode* pNext = pNode->next;
            if(pNext == nullptr)
                pReversedHead = pNode;
            pNode->next = pPrev;
            pPrev = pNode;
            pNode = pNext;
        }
        return pReversedHead;
    }
};


```

```python
class Solution:
    # 返回ListNode
    def ReverseList(self, pHead):
        # write code here
        pReversedHead, pNode, pRrev = None, pHead, None
        while pNode != None:
            pNext = pNode.next
            if pNext == None:
                pReversedHead = pNode
            pNode.next = pRrev
            pRrev = pNode
            pNode = pNext
        return pReversedHead


```
