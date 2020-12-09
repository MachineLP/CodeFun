```
题目描述

给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。例如，如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，那么一共存在6个滑动窗口，他们的最大值分别为{4,4,6,6,6,5}； 针对数组{2,3,4,2,6,2,5,1}的滑动窗口有以下6个： {[2,3,4],2,6,2,5,1}， {2,[3,4,2],6,2,5,1}， {2,3,[4,2,6],2,5,1}， {2,3,4,[2,6,2],5,1}， {2,3,4,2,[6,2,5],1}， {2,3,4,2,6,[2,5,1]}。

解题思路

只将可能成为滑动窗口最大值的数值存入一个双端队列，即该双端队列中数字为非递增序列，队列的队头元素即为最大值。

以{2,3,4,2,6,2,5,1}，size=3为例。

初始化窗口：2进入队列，遍历3，则2不可能为最大值，压入3，弹出2，同理遍历4，则3不可能为最大值，压入4，弹出3，初始化完成。

遍历2：压入队列后满足非递增，直接压入2

遍历6：压入后不满足非递增，依次从队尾弹出2,4，压入6

遍历2：压入队列后满足非递增，直接压入2

遍历5：压入后不满足非递增，从队尾弹出2，压入5

遍历1：压入后满足非递增，压入1，同时此时由于6不在窗口内，从队头弹出6

```


```C++
class Solution {
public:
    vector<int> maxInWindows(const vector<int>& num, unsigned int size)
    {
        vector<int>ans;
        if(size < 1 || num.size() < size)
            return ans;
        deque<int> index;
        for(unsigned int i = 0; i < size; ++ i){
            while(!index.empty() && num[i] >= num[index.back()])
                index.pop_back();
            index.push_back(i);
        }
        for(unsigned int i = size; i < num.size(); ++ i){
            ans.push_back(num[index.front()]);
            while(!index.empty() && num[i] >= num[index.back()])
                index.pop_back();
            if(!index.empty() && index.front() <= int(i - size))
                index.pop_front();
            index.push_back(i);
        }
        ans.push_back(num[index.front()]);
        return ans;
    }
};


```


```python
class Solution:
    def maxInWindows(self, num, size):
        # write code here
        if size == 0 or size > len(num):
            return []
        myqueue, ans = [], []
        for i in range(size):
            while len(myqueue) and myqueue[-1] < num[i]:
                del myqueue[-1]
            myqueue.append(num[i])
        ans.append(myqueue[0])
        for i in range(size, len(num)):
            while len(myqueue) and myqueue[-1] < num[i]:
                del myqueue[-1]
            myqueue.append(num[i])
            if myqueue[0] == num[i-size]:
                del myqueue[0]
            ans.append(myqueue[0])
        return ans


```
