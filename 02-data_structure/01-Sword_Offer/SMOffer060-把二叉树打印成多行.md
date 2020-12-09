```
题目描述

从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行。

解题思路

本题其实没有59题难，用队列保存将要打印的节点，每次遍历到该节点的时候，将其弹出，并将其左右孩子（若存在）弹入队列。不过为了区分出每一层的分隔，设一个last记录每一层的最后一个节点，当遍历到该节点，说明此层遍历结束，更新last为当前队列的队尾值。

```


```C++
class Solution {
public:
        vector<vector<int> > Print(TreeNode* pRoot) {
            vector<vector<int>>ans;
            int level = 0;
            TreeNode* last = pRoot;
            queue<TreeNode*>q;
            if(pRoot != nullptr)
                q.push(pRoot);
            ans.push_back({});
            while(!q.empty()){
                TreeNode* temp = q.front();
                if(temp->left != nullptr)
                    q.push(temp->left);
                if(temp->right != nullptr)
                    q.push(temp->right);
                ans[level].push_back(temp->val);
                if(last == temp){
                    last = q.back();
                    ++ level;
                    ans.push_back({});
                }
                q.pop();
            }
            ans.pop_back();
            return ans;
        }
};


```


```python
class Solution:
    # 返回二维列表[[1,2],[4,5]]
    def Print(self, pRoot):
        # write code here
        if pRoot == None:
            return []
        ans, tempans, last = [], [], pRoot
        q = [pRoot]
        while len(q):
            temp = q[0]
            if temp.left:
                q.append(temp.left)
            if temp.right:
                q.append(temp.right)
            tempans.append(temp.val)
            if last == temp:
                ans.append(tempans)
                tempans = []
                last = q[-1]
            del q[0]
        return ans


```
