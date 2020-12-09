```
题目描述

输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），返回结果为复制后复杂链表的head。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）

解题思路

首先复制每个节点N，N'位于N的后面，其次复制每个节点的random，对每个节点N而言，假设原始的random指向S，则复制节点N’就指向S‘，而S的next节点指向的就是S‘，所以可以直接复制节点的random节点。最后拆分链表，偶数位的节点即为我们要求的节点。

```

```C++
class Solution {
public:
    RandomListNode* Clone(RandomListNode* pHead)
    {
        CloneNode(pHead);
        CloneRand(pHead);
        RandomListNode *p = pHead, *pClone = nullptr, *q;
        if(p != nullptr){
            pClone = p->next;
            q = pClone;
            p->next = pClone->next;
            p = p->next;
        }
        while(p != nullptr){
            q->next = p->next;
            q = q->next;
            p->next = q->next;
            p = p->next;
        }
        return pClone;
    }
    void CloneNode(RandomListNode* pHead){
        RandomListNode* p = pHead;
        while(p != nullptr){
            RandomListNode* q = new RandomListNode(p->label);
            q->next = p->next;
            p->next = q;
            p = q->next;
        }
    }
    void CloneRand(RandomListNode* pHead){
        RandomListNode* p = pHead, *q;
        while(p != nullptr){
            q = p->next;
            if(p->random != nullptr)
                q->random = p->random->next;
            p = q->next;
        }
    }
};


```

```python
class Solution:
    # 返回 RandomListNode
    def Clone(self, pHead):
        # write code here
        if pHead == None:
            return None
        pNode = pHead
        while pNode:
            pClone = RandomListNode(pNode.label)
            pClone.next = pNode.next
            pNode.next = pClone
            pNode = pClone.next
        pNode = pHead
        while pNode:
            pClone = pNode.next
            if pNode.random:
                pClone.random = pNode.random.next
            pNode = pClone.next
        pCur, pCopy = pHead, pHead.next
        while pCur:
            pClone = pCur.next
            pCur.next = pClone.next
            if pCur.next:
                pClone.next = pCur.next.next
            pCur = pCur.next
        return pCopy


```
