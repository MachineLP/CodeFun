```
题目描述

输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。

解题思路

常规题型，根据二叉树的前序，中序结果可以完全确定一棵树，根据前序遍历的第一个数字创建根节点，在中序遍历中找到根节点的位置，由该位置可以划分右子树和左子树的序列，接着递归调用函数构建左右子树

```


```C++
class Solution {
public:
    TreeNode* reConstructBinaryTree(vector<int> pre,vector<int> vin) {
        if(pre.size() == 0)
            return nullptr;
        vector<int>preLeft, preRight, inLeft, inRight;
        TreeNode *head = new TreeNode(pre[0]);
        int root = 0;
        for(int i = 0; i < pre.size(); ++ i)
            if(pre[0] == vin[i]){
                root = i;
                break;
            }
        for(int i = 0; i < root; ++ i){
            inLeft.push_back(vin[i]);
            preLeft.push_back(pre[i+1]);
        }
        for(int i = root + 1; i < pre.size(); ++ i){
            inRight.push_back(vin[i]);
            preRight.push_back(pre[i]);
        }
        head->left = reConstructBinaryTree(preLeft, inLeft);
        head->right = reConstructBinaryTree(preRight, inRight);
        return head;
    }
};


```

```python
class Solution:
    # 返回构造的TreeNode根节点
    def reConstructBinaryTree(self, pre, tin):
        # write code here
        if len(pre) == 0:
            return None
        elif len(pre) == 1:
            return TreeNode(pre[0])
        root = TreeNode(pre[0])
        pos = tin.index(pre[0])
        root.left = self.reConstructBinaryTree(pre[1:pos+1], tin[:pos])
        root.right = self.reConstructBinaryTree(pre[pos+1:], tin[pos+1:])
        return root

```
