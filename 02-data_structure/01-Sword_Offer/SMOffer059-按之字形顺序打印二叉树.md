```
题目描述

请实现一个函数按照之字形打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右至左的顺序打印，第三行按照从左到右的顺序打印，其他行以此类推。

解题思路

在打印某一层节点时，把下一层的子节点保存在相应的栈里，若当前打印的是奇数层，则先保存右子节点，再保存左子节点；若当前打印的是偶数层，则先保存左子节点，再保存右子节点。从第0层开始计数。

```


```C++
class Solution {
public:
    vector<vector<int> > Print(TreeNode* pRoot) {
        int level = 0;
        TreeNode *temptail = pRoot;
        vector<vector<int>>ans;
        if(pRoot == nullptr)
            return ans;
        stack<TreeNode*>s[2];
        s[0].push(pRoot);
        ans.resize(1);
        while(!s[0].empty() || !s[1].empty()){
            int tempstack = level % 2;
            TreeNode *temp = s[tempstack].top();
            if(tempstack == 0){
                if(temp->left != nullptr)
                    s[1].push(temp->left);
                if(temp->right != nullptr)
                    s[1].push(temp->right);
            }
            else{
                if(temp->right != nullptr)
                    s[0].push(temp->right);
                if(temp->left != nullptr)
                    s[0].push(temp->left);
            }
            ans[level].push_back(temp->val);
            s[tempstack].pop();
            if(s[tempstack].empty()){
                level ++;
                ans.resize(level + 1);
            }
        }
        ans.resize(level);
        return ans;
    }
};


```

```python
class Solution:
    def Print(self, pRoot):
        # write code here
        if pRoot == None:
            return []
        ans, tempans, tempstack = [],[], [[pRoot],[]]
        level = 0
        while len(tempstack[0]) > 0 or len(tempstack[1]) > 0:
            stackid = level % 2
            temp = tempstack[stackid][-1]
            if stackid == 0:
                if temp.left != None:
                    tempstack[1].append(temp.left)
                if temp.right != None:
                    tempstack[1].append(temp.right)
            else:
                if temp.right != None:
                    tempstack[0].append(temp.right)
                if temp.left != None:
                    tempstack[0].append(temp.left)
            tempans.append(temp.val)
            del tempstack[stackid][-1]
            if len(tempstack[stackid]) == 0:
                level += 1
                ans.append(tempans)
                tempans = []
        return ans


```
