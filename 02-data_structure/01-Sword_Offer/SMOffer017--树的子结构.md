```
题目描述

输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）

解题思路

递归调用HasSubstree遍历二叉树Proot1，如果发现某一节点的值与PRoot2的根节点值相同，则调用DotreeSame判断pRoot1

中以该节点为根节点的子树与pRoot2是否相同。

```

```C++
class Solution {
public:
    bool HasSubtree(TreeNode* pRoot1, TreeNode* pRoot2)
    {
        bool result = false;
        if(pRoot1 != nullptr && pRoot2 != nullptr){
            if(pRoot1->val == pRoot2->val)
                result = DoTreeSame(pRoot1, pRoot2);
            if(!result)
                result = HasSubtree(pRoot1->left, pRoot2);
            if(!result)
                result = HasSubtree(pRoot1->right, pRoot2);
        }
        return result;
    }
    bool DoTreeSame(TreeNode* pRoot1, TreeNode* pRoot2){
        if(pRoot2 == nullptr)
            return true;
        if(pRoot1 == nullptr)
            return false;
        if(pRoot1->val != pRoot2->val)
            return false;
        return DoTreeSame(pRoot1->left,pRoot2->left) && DoTreeSame(pRoot1->right, pRoot2->right);
    }
};


```


```python
class Solution:
    def HasSubtree(self, pRoot1, pRoot2):
        # write code here
        res = False
        if pRoot1 != None and pRoot2 != None:
            if pRoot1.val == pRoot2.val:
                res = self.DoesTree1HaveTree2(pRoot1, pRoot2)
            if res == False:
                res = self.HasSubtree(pRoot1.left, pRoot2)
            if res == False:
                res = self.HasSubtree(pRoot1.right, pRoot2)
        return res
    def DoesTree1HaveTree2(self, pRoot1, pRoot2):
        if pRoot2 == None:
            return True
        if pRoot1 == None:
            return False
        if pRoot1.val != pRoot2.val:
            return False
        return self.DoesTree1HaveTree2(pRoot1.left, pRoot2.left) and self.DoesTree1HaveTree2(pRoot1.right, pRoot2.right)


```
