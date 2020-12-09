```
题目描述

请实现一个函数，用来判断一颗二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。

解题思路

前序遍历的顺序为DLR，定义一种对称前序遍历为DRL，通过比较这两种遍历方式可以判断是否对称


```


```C++
class Solution {
public:
    bool isSymmetrical(TreeNode* pRoot)
    {
        return pRoot == nullptr ? true : isSymmtrical(pRoot, pRoot);
    }
    bool isSymmtrical(TreeNode* pRoot1, TreeNode* pRoot2){
        if(pRoot1 == nullptr && pRoot2 == nullptr)
            return true;
        if(pRoot1 == nullptr || pRoot2 == nullptr)
            return false;
        if(pRoot1->val != pRoot2->val)
            return false;
        return isSymmtrical(pRoot1->left, pRoot2->right) && isSymmtrical(pRoot1->right, pRoot2->left);
    }
};


```

```python
class Solution:
    def isSymmetrical(self, pRoot):
        # write code here
        return True if pRoot == None else self.myIsSymmtrical(pRoot.left, pRoot.right)
    def myIsSymmtrical(self, pRoot1, pRoot2):
        if pRoot1 == None and pRoot2 == None:
            return True
        if pRoot1 == None or pRoot2 == None or pRoot1.val != pRoot2.val:
            return False
        return self.myIsSymmtrical(pRoot1.left, pRoot2.right) and self.myIsSymmtrical(pRoot1.right, pRoot2.left)


```
