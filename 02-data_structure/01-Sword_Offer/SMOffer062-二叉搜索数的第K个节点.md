```
题目描述

给定一棵二叉搜索树，请找出其中的第k小的结点。例如， （5，3，7，2，4，6，8）    中，按结点数值大小顺序第三小结点的值为4。

解题思路

中序遍历的第k个节点即为我们要求的节点
```


```C++
class Solution {
public:
    TreeNode* ans;
    int cnt;
    TreeNode* KthNode(TreeNode* pRoot, int k)
    {
        ans = nullptr;
        cnt = 0;
        intravel(pRoot, k);
        return ans;
    }
 
    void intravel(TreeNode* root, int k){
        if(cnt > k || root == nullptr)
            return;
        intravel(root->left, k);
        if(++ cnt == k){
            ans = root;
            return ;
        }  
        intravel(root->right, k);
    }
};


```


```python
class Solution:
    # 返回对应节点TreeNode
    def __init__(self):
        self.ans = None
        self.cnt = 0
    def KthNode(self, pRoot, k):
        # write code here
        if pRoot == None:
            return None
        self.myKthNode(pRoot, k)
        return self.ans
    def myKthNode(self, pRoot, k):
        if self.cnt > k:
            return
        if pRoot.left:
            self.myKthNode(pRoot.left, k)
        self.cnt += 1
        if self.cnt == k:
            self.ans = pRoot
        if pRoot.right:
            self.myKthNode(pRoot.right, k)


```
