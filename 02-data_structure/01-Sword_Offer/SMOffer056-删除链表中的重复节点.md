```
题目描述

在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。 例如，链表1->2->3->3->4->4->5 处理后为 1->2->5

解题思路

遍历链表，我们要确保当前遍历到的节点始终与下一个不重复的节点连在一起。

pPre记录上一个不重复的节点，pCur为当前遍历到的节点，pNext临时用来检查pCur是否重复。

```


```C++
class Solution {
public:
    ListNode* deleteDuplication(ListNode* pHead)
    {
        if(pHead == nullptr)
            return nullptr;
        ListNode *pPre = nullptr, *pCur = pHead, *pNext = nullptr;
        while(pCur != nullptr){
            int cnt = 0, val = pCur->val;
            pNext = pCur->next;
            while(pNext != nullptr && pNext->val == val){
                pNext = pNext->next;
                ++ cnt;
            }
            if(cnt > 0){
                if(pPre == nullptr)
                    pHead = pNext;
                else
                    pPre->next = pNext;
            }
            else
                pPre = pCur;
            pCur = pNext;
        }
        return pHead;
    }
};


```


```python
class Solution:
    def deleteDuplication(self, pHead):
        # write code here
        if pHead == None:
            return None
        pPre, pCur, pNext = None, pHead, None
        while pCur != None:
            pNext = pCur.next
            cnt = 0
            while pNext != None and pNext.val == pCur.val:
                pNext = pNext.next
                cnt += 1
            if cnt:
                if pPre == None:
                    pHead = pNext
                else:
                    pPre.next = pNext
            else:
                pPre = pCur
            pCur = pNext
        return pHead


```
