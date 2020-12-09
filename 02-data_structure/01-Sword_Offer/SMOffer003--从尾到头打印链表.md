```
题目描述

输入一个链表，按链表值从尾到头的顺序返回一个ArrayList。

解题思路

没什么特别的算法，考察栈的使用，或者说用栈将递归函数改为非递归。
```

```C++
class Solution {
public:
    vector<int> printListFromTailToHead(ListNode* head) {
        stack<int>nodes;
        vector<int>ans;
        ListNode *node = head;
        while(node != nullptr){
            nodes.push(node->val);
            node = node->next;
        }
        while(!nodes.empty()){
            ans.push_back(nodes.top());
            nodes.pop();
        }
        return ans;
    }
};
 
//直接在vector的前端插入
class Solution {
public:
    vector<int> printListFromTailToHead(ListNode* head) {
        vector<int>ans;
        ListNode *node = head;
        while(node != nullptr){
            ans.insert(ans.begin(), node->val);
            node = node->next;
        }
        return ans;
    }
};

```

```python
class Solution:
    # 返回从尾部到头部的列表值序列，例如[1,2,3]
    def printListFromTailToHead(self, listNode):
        # write code here
        nodes = []
        ans = []
        while listNode:
            nodes.append(listNode.val)
            listNode = listNode.next
        for i in range(len(nodes)-1,-1,-1):
            ans.append(nodes[i])
        return ans
 
#直接用insert
class Solution:
    # 返回从尾部到头部的列表值序列，例如[1,2,3]
    def printListFromTailToHead(self, listNode):
        # write code here
        ans = []
        while listNode:
            ans.insert(0, listNode.val)
            listNode = listNode.next
        return ans

```
