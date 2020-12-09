```
题目描述

用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。

解题思路

考察点是栈和队列的基本概念，栈是先进后出，队列先进先出，有两个栈的话，一个栈stack1作为输入，一个栈stack2作为输出。

输入：直接向stack1中压入元素

输出：如果stack2不空，直接将stack2的栈顶弹出；若为空，需要将stack1中的元素依次从栈顶开始压入stack2。

```

```C++
class Solution
{
public:
    void push(int node) {
        stack1.push(node);
    }
    int pop() {
        int temp, ans;
        if(stack2.size() == 0){
            while(!stack1.empty()){
                temp = stack1.top();
                stack1.pop();
                stack2.push(temp);
            }
        }
        ans = stack2.top();
        stack2.pop();
        return ans;
    }
private:
    stack<int> stack1;
    stack<int> stack2;
};

```

```python
class Solution:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []
    def push(self, node):
        # write code here
        self.stack1.append(node)
    def pop(self):
        # return xx
        if len(self.stack2) == 0:
            while len(self.stack1) > 0:
                self.stack2.append(self.stack1.pop())
        return self.stack2.pop()

```
