```
题目描述

给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。

解题思路

分几种情况讨论：

①若该节点的右孩子不为空，则中序遍历的下一个节点为右子树的最左节点

②右孩子为空，该节点为其父节点的左孩子，则下一个节点为该父节点

③右孩子为空，该节点为其父节点的左孩子则沿着其父节点一直向上遍历，知道找到某节点为该节点的父节点的左子树

```


```C++
class Solution {
public:
    TreeLinkNode* GetNext(TreeLinkNode* pNode)
    {
        if(pNode == nullptr)
            return nullptr;
        TreeLinkNode *pNext = nullptr;
        if(pNode->right != nullptr){
            TreeLinkNode *pRight = pNode->right;
            while(pRight->left != nullptr)
                pRight = pRight->left;
            pNext = pRight;
        } 
        else if(pNode->next != nullptr){
            TreeLinkNode *pParent = pNode->next, *pCur = pNode;
            while(pParent != nullptr && pParent->right == pCur){
                pCur = pParent;
                pParent = pParent->next;
            }
            pNext = pParent;
        }
        return pNext;
    }
};


```


```python
class Solution:
    def GetNext(self, pNode):
        # write code here
        if pNode == None:
            return None
        pNext = None
        if pNode.right != None:
            pNext = pNode.right
            while pNext.left != None:
                pNext = pNext.left
        elif pNode.next != None:
            pNext, pCur = pNode.next, pNode
            while pNext != None and pNext.right == pCur:
                pCur = pNext
                pNext = pCur.next
        return pNext


```
