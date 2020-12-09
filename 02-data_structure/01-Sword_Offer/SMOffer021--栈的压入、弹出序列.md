```
输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）

解题思路

本题为栈的简单模拟，建立一个辅助栈，将输入的第一个序列中的数字依次压入辅助栈，并按照第二个序列的顺序依次从该栈中弹出数字

```

```C++
class Solution {
public:
    bool IsPopOrder(vector<int> pushV,vector<int> popV) {
        if(pushV.size() > 0){
            int i = 0, j = 0;
            stack<int>s;
            while(j != popV.size()){
                while(s.empty() || s.top() != popV[j]){
                    if(i == pushV.size())
                        break;
                    s.push(pushV[i++]);
                }
                if(s.top() != popV[j++])
                    break;
                s.pop();
                if(i == pushV.size() && j == popV.size())
                    return true;
            }
        }
        return false;
    }
};


```

```python
class Solution:
    def IsPopOrder(self, pushV, popV):
        # write code here
        i, j, Len = 0, 0, len(pushV)
        tempStack = []
        while j < Len:
            while len(tempStack) == 0 or tempStack[-1] != popV[j]:
                if i == Len:
                    break
                tempStack.append(pushV[i])
                i += 1
            if popV[j] != tempStack[-1]:
                break;
            tempStack.pop()
            j += 1
            if i == Len and j == Len:
                return True
        return False


```
