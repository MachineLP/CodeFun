```
题目描述

输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。

解题思路

BST的特点是其中序遍历是排好序的，换言之，最终的双向链表顺序即为二叉搜索树的中序遍历。按照中序遍历的顺序，当我们遍历转换到根节点时，其左子树已经转换成一个排序的链表，且链表的最后一个节点为左子树最大的节点，将根节点与最后一个节点链接起来，接着开始遍历转换右子树。

```

```C++
class Solution {
public:
    TreeNode* Convert(TreeNode* pRootOfTree)
    {
        TreeNode* pLast = nullptr;
        ConvertNode(pRootOfTree, &pLast);
        while(pRootOfTree != nullptr && pRootOfTree->left != nullptr)
            pRootOfTree = pRootOfTree->left;
        return pRootOfTree;
    }
    void ConvertNode(TreeNode* pNode, TreeNode** pLast){
        if(pNode == nullptr)
            return;
        TreeNode* pCur = pNode;
        if(pCur->left != nullptr)
            ConvertNode(pCur->left, pLast);
        pCur->left = *pLast;
        if(*pLast != nullptr)
            (*pLast)->right = pCur;
        *pLast = pCur;
        if(pCur->right != nullptr)
            ConvertNode(pCur->right, pLast);
    }
};


```

```python
class Solution:
    def __init__(self):
        self.pLast = None
    def Convert(self, pRootOfTree):
        # write code here
        if pRootOfTree == None:
            return None
        self.ConvertNode(pRootOfTree)
        while pRootOfTree and pRootOfTree.left:
            pRootOfTree = pRootOfTree.left
        return pRootOfTree
    def ConvertNode(self, pRootOfTree):
        pCur = pRootOfTree
        if pCur.left:
            self.ConvertNode(pCur.left)
        pCur.left = self.pLast
        if self.pLast != None:
            self.pLast.right = pCur
        self.pLast = pCur
        if pCur.right:
            self.ConvertNode(pCur.right)


```
