```
题目描述

输入一颗二叉树的跟节点和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。(注意: 在返回值的list中，数组长度大的数组靠前)

解题思路

用前序遍历的方式访问每个节点，将该节点添加到路径上，并累加该节点的值，若该节点为叶子节点，则比较当前节点值之和是否等于目标值，若相等，则将该路径存储，若该节点不是叶子节点，则继续访问它的叶子节点

```

```C++
class Solution {
public:
    vector<vector<int> > FindPath(TreeNode* root,int expectNumber) {
        ans.clear();
        temppath.clear();
        if(root == nullptr)
            return ans;
        FindPathPart(root, expectNumber, root->val);
        return ans;
    }
private:
    vector<vector<int>>ans;
    vector<int>temppath;
    void FindPathPart(TreeNode* root, int target, int sum){
        if(target < sum)
            return;
        temppath.push_back(root->val);
        if(root->left == nullptr && root->right == nullptr && target == sum)
            ans.push_back(temppath);
        if(root->left != nullptr)
            FindPathPart(root->left, target, sum + root->left->val);
        if(root->right != nullptr)
            FindPathPart(root->right, target, sum + root->right->val);
        temppath.pop_back();
    }
};


```

```python
class Solution:
    # 返回二维列表，内部每个列表表示找到的路径
    def __init__(self):
        self.ans = []
        self.temppath = []
        self.sum = 0
    def FindPath(self, root, expectNumber):
        # write code here
        if root == None:
            return []
        self.sum += root.val
        self.temppath.append(root.val)
        if root.left == None and root.right == None and self.sum == expectNumber:
            self.ans.append(self.temppath[:])
        if root.left != None:
            self.FindPath(root.left, expectNumber)
        if root.right != None:
            self.FindPath(root.right, expectNumber)
        self.sum -= root.val
        self.temppath.pop()
        return self.ans


```
