```
题目描述

给一个链表，若其中包含环，请找出该链表的环的入口结点，否则，输出null。

解题思路

本题可以分解以下几步：

首先判断是否存在环：双指针法，一快一慢两个指针，若两个指针可以相遇，说明有环存在。

其次计算环中节点的个数：在上述相遇位置慢继续遍历，当再次到达这个位置的时候即可得到环中节点的个数n。

最后计算链表倒数第n个节点：双指针，先让一个指针遍历n，再两个指针一起遍历，与常见题型不同的是本题的判断条件不是第一个指针为空，而是第一个指针与第二个指针相遇。

```


```C++
class Solution {
public:
    ListNode* MeetingNode(ListNode* pHead){
        if(pHead == nullptr)
            return nullptr;
        ListNode *p = pHead->next, *q = pHead;
        while(q != nullptr && p != nullptr){
            q = q->next;
            if(q == nullptr)
                return nullptr;
            else
                q = q->next;
            if(p == q)
                return p;
            p = p->next;
        }
        return nullptr;
    }
    ListNode* EntryNodeOfLoop(ListNode* pHead)
    {
        ListNode* meetingNode = MeetingNode(pHead);
        if(meetingNode == nullptr)
            return nullptr;
        int nodesInLoop = 1;
        ListNode *p = meetingNode, *q = pHead;
        while(p->next != meetingNode){
            p = p->next;
            ++ nodesInLoop;
        }
        p = pHead;
        for(int i = 0; i < nodesInLoop; ++ i)
            p = p->next;
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
    def EntryNodeOfLoop(self, pHead):
        # write code here
        meetingNode = self.MeetingNode(pHead)
        if meetingNode == None:
            return None
        p, cnt = meetingNode.next, 1
        while p != meetingNode:
            p = p.next
            cnt += 1
        p, q = pHead, pHead
        for i in range(cnt):
            p = p.next
        while p != q:
            p = p.next
            q = q.next
        return p
    def MeetingNode(self, pHead):
        if pHead == None or pHead.next == None:
            return None
        pSlow, pFast = pHead.next, pHead.next.next
        while pFast != None:
            if pFast == pSlow:
                return pFast
            pSlow = pSlow.next
            pFast = pFast.next
            if pFast.next != None:
                pFast = pFast.next
        return None


```
