```
题目描述

请实现两个函数，分别用来序列化和反序列化二叉树

解题思路

序列化格式：根据前序遍历，从根节点开始，nullptr记为$

序列化：递归，遍历二叉树，若遍历的节点为空节点，则记为$，否则记录其节点的值，并递归遍历他的左孩子和右孩子。

反序列化：递归，以“1,2,4,$,$,$,3,5,$,$,6,$,$”，第一个读出的数字为1，这是根节点，第二个为2，1的左孩子，第三个为4,2的左孩子，第四和第五均为$，说明4的左右孩子为空，接着回到2节点，由此类推。

```

```C++
class Solution {
public:
    char* Serialize(TreeNode *root) {    
        if(root == nullptr)
            return nullptr;
        string str;
        SerializeCore(root, str);
        int len = str.length();
        char* res = new char[len + 1];
        str.copy(res, str.length(), 0);
        return res;
    }
    TreeNode* Deserialize(char *str) {
        if(str == nullptr)
            return nullptr;
        TreeNode* res = DeserializeCore(&str);
        return res;
    }
    void SerializeCore(TreeNode* root, string& str){
        if(root == nullptr){
            str += '#';
            return;
        }
        string temp = to_string(root->val);
        str += temp;
        str += ',';
        SerializeCore(root->left, str);
        SerializeCore(root->right, str);
    }
    TreeNode* DeserializeCore(char** str){
        if(**str == '#'){
            (*str)++;
            return nullptr;
        }
        int num = 0;
        while(**str != ',' && **str != '\0'){
            num = num * 10 + ((**str) - '0');
            (*str) ++;
        }
        TreeNode* root = new TreeNode(num);
        if(**str == '\0')
            return root;
        else
            (*str)++;
        root->left = DeserializeCore(str);
        root->right = DeserializeCore(str);
        return root;
    }
};


```


```python
class Solution:
    def __init__(self):
        self.s = ""
        self.cur = 0
    def Serialize(self, root):
        # write code here
        if root == None:
            return "#"
        self.s = str(root.val)
        self.mySerialize(root.left)
        self.mySerialize(root.right)
        return self.s
    def mySerialize(self, root):
        if root == None:
            self.s += ",#"
            return;
        self.s += ',' + str(root.val)
        self.mySerialize(root.left)
        self.mySerialize(root.right)
    def Deserialize(self, s):
        # write code here
        if s == None:
            return None
        value = s.split(',')
        return self.myDeserialize(value)
    def myDeserialize(self, value):
        if self.cur == len(value):
            return
        if value[self.cur] == '#':
            self.cur += 1
            return
        Root = TreeNode(int(value[self.cur]))
        self.cur += 1
        Root.left = self.myDeserialize(value)
        Root.right = self.myDeserialize(value)
        return Root


```

