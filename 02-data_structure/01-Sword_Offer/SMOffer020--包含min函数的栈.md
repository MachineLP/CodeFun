```
题目描述

定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O（1））。

解题思路

定义两个栈，一个用来记录所有数据，另一个记录最小元素，当压入栈的时候，比较最小栈的栈顶元素是否大于待压入值，若大于等于则压入栈，在出栈的时候，若出栈元素等于最小栈的栈顶元素，则需要将最小栈栈顶一并出栈。

```

```C++
class Solution {
public:
    void push(int value) {
        Data.push(value);
        if(Min.empty() || Min.top() >= value)
            Min.push(value);
    }
    void pop() {
        if(Data.top() == Min.top())
            Min.pop();
        Data.pop();
    }
    int top() {
        return Data.top();
    }
    int min() {
        return Min.top();
    }
    private:
        stack<int>Data, Min;
};


```

```python
class Solution:
    def __init__(self):
        self.Min, self.Data = [], []
    def push(self, node):
        # write code here
        self.Data.append(node)
        if len(self.Min) == 0 or self.Min[-1] >= node:
            self.Min.append(node)
    def pop(self):
        # write code here
        node = self.Data.pop()
        if len(self.Min) > 0 and self.Min[-1] == node:
            self.Min.pop()
    def top(self):
        # write code here
        return self.Data[-1]
    def min(self):
        # write code here
        return self.Min[-1]


```
