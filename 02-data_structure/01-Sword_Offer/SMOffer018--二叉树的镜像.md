```
题目描述

操作给定的二叉树，将其变换为源二叉树的镜像。

输入描述:

二叉树的镜像定义：源二叉树 
    	    8
    	   /  \
    	  6   10
    	 / \  / \
    	5  7 9 11
    	镜像二叉树
    	    8
    	   /  \
    	  10   6
    	 / \  / \
    	11 9 7  5
解题思路

前序遍历这棵树的每个节点，如果遍历到的节点有子节点，则交换它的两个子节点。

```

```C++
class Solution {
public:
    void Mirror(TreeNode *pRoot) {
        if(pRoot == nullptr)
            return;
        if(pRoot->left == nullptr && pRoot->right == nullptr)
            return;
        TreeNode *temp = pRoot->left;
        pRoot->left = pRoot->right;
        pRoot->right = temp;
        Mirror(pRoot->left);
        Mirror(pRoot->right);
    }
};


```

```python
class Solution:
    # 返回镜像树的根节点
    def Mirror(self, root):
        # write code here
        if root != None:
            if root.left != None or root.right != None:
                temp = root.left
                root.left = root.right
                root.right = temp
                self.Mirror(root.left)
                self.Mirror(root.right)

```
