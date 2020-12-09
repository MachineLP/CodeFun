```
题目描述

输入一棵二叉树，判断该二叉树是否是平衡二叉树。

解题思路

本题只用考虑各个节点的左右子树的高度是否满足AVL，不需要考虑左右节点的大小。按照最常规的思路，对每个节点计算其左右子树的高度，继而进行判断，但是这样会导致重复运算。

所以我们在遍历某节点的左右子节点之后，可以根据它的左右子节点的深度判断它是不是平衡的，并得到当前的深度，当最后遍历到树的根节点的时候，也就判断整棵树的平衡。

```


```C++
class Solution {
public:
    bool IsBalanced_Solution(TreeNode* pRoot) {
        if(pRoot == nullptr)
            return true;
        int Height = 0;
        return IsBalanced(pRoot, &Height);
    }
    bool IsBalanced(TreeNode* pRoot, int* Height){
        if(pRoot == nullptr){
            *Height = 0;
            return true;
        }
        int left, right;
        if(IsBalanced(pRoot->left, &left) && IsBalanced(pRoot->right, &right)){
            int diff = left - right;
            if(diff <= 1 && diff >= -1){
                *Height = 1 + max(left, right);
                return true;
            }
        }
        return false;
    }
};


```


```python
class Solution:
    def __init__(self):
        self.flag = True
    def IsBalanced_Solution(self, pRoot):
        # write code here
        self.IsBalanced(pRoot)
        return self.flag
    def IsBalanced(self, pRoot):
        if self.flag == False or pRoot == None:
            return 0;
        left = self.IsBalanced(pRoot.left)
        right = self.IsBalanced(pRoot.right)
        if abs(left - right) > 1:
            self.flag = False
        return max(left, right) + 1


```
