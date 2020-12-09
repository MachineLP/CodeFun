```
题目描述

从上往下打印出二叉树的每个节点，同层节点从左至右打印。

解题思路

二叉树的层次遍历
```

```C++
class Solution {
public:
    vector<int> PrintFromTopToBottom(TreeNode* root) {
        if(root == nullptr)
            return ans;
        queue<TreeNode*>q;
        q.push(root);
        while(!q.empty()){
            TreeNode* temp = q.front();
            q.pop();
            ans.push_back(temp->val);
            if(temp->left != nullptr)
                q.push(temp->left);
            if(temp->right != nullptr)
                q.push(temp->right);
        }
        return ans;
    }
    private:
        vector<int>ans;
};


```

```python
class Solution:
    # 返回从上到下每个节点值列表，例：[1,2,3]
    def PrintFromTopToBottom(self, root):
        # write code here
        ans, tempQueue = [], []
        if root == None:
            return ans
        tempQueue.append(root)
        while len(tempQueue):
            temp = tempQueue.pop(0)
            ans.append(temp.val)
            if temp.left != None:
                tempQueue.append(temp.left)
            if temp.right != None:
                tempQueue.append(temp.right)
        return ans


```
