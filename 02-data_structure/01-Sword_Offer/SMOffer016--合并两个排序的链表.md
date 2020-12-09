```
题目描述

输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。

解题思路

采用递归的方法，假设待合并的链表1，链表2，目前链表1头结点值小于链表2，则链表1的头结点为合并后链表的链表的头结点，则合并后的链表头节点指向的next的值为链表1的剩余节点与链表2的合并后的链表

```

```C++
class Solution {
public:
    ListNode* Merge(ListNode* pHead1, ListNode* pHead2)
    {
        if(pHead1 == nullptr)
            return pHead2;
        if(pHead2 == nullptr)
            return pHead1;
        ListNode *pAns = nullptr;
        if(pHead1->val < pHead2->val){
            pAns = pHead1;
            pAns->next = Merge(pHead1->next, pHead2);
        }
        else{
            pAns = pHead2;
            pAns->next = Merge(pHead1, pHead2->next);
        }
        return pAns;
    }
};


```

```python
class Solution:
    # 返回合并后列表
    def Merge(self, pHead1, pHead2):
        # write code here
        if pHead1 == None:
            return pHead2
        if pHead2 == None:
            return pHead1
        pAns = None
        if pHead1.val < pHead2.val:
            pAns = pHead1
            pAns.next = self.Merge(pHead1.next, pHead2)
        else:
            pAns = pHead2
            pAns.next = self.Merge(pHead1, pHead2.next)
        return pAns


```
