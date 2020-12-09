```
题目描述

输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。

解题思路

二叉搜索树有一个特点，对任意节点而言，左子树的节点值都比该节点值小，右子树的节点值都比该节点值大。所以后序遍历得到的序列中，最后一个数字是树的根节点的值，前面的数字可以分为两部分，第一部分为左子树的值，都比根节点小，第二部分为右子树的值，都比根节点大。

```

```C++
class Solution {
public:
    bool VerifySquenceOfBST(vector<int> sequence) {
        int len = sequence.size();
        if(len == 0)
            return false;
        return PartVerifySquenceOfBST(sequence, 0, len - 1);
    }
    bool PartVerifySquenceOfBST(vector<int>seq, int left, int right){
        if(right <= left)
            return true;
        int root = seq[right], i = left, j;
        bool flag = true;
        while(seq[i] < root)
            ++ i;
        j = i;
        while(i < right)
            if(seq[++ i] < root)
                return false;
        flag = PartVerifySquenceOfBST(seq, left, i - 1) && PartVerifySquenceOfBST(seq, i, right - 1);
        return flag;
    }
};


```

```python
class Solution:
    def VerifySquenceOfBST(self, sequence):
        # write code here
        if len(sequence) == 0:
            return False
        if len(sequence) == 1:
            return True
        root,i,Len = sequence[-1],0,len(sequence)
        while sequence[i] < root:
            i += 1
        j = i
        while i < Len and sequence[i] >= root:
            i += 1
        if i != Len:
            return False
        Left, Right = True, True
        if j > 1:
            Left = self.VerifySquenceOfBST(sequence[0:j-1])
        if Len - j - 1 > 0:
            Right = self.VerifySquenceOfBST(sequence[j: Len-1])
        return Left and Right


```
