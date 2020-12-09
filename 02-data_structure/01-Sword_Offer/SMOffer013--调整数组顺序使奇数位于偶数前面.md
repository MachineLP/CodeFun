```
题目描述

输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。

解题思路

本题与原书上的意思还是有点差别的，原书不需要保证奇数和奇数，偶数和偶数之间的相对位置不变，所以可以用快排进行一次划分的时候采用的双指针法。但是这里需要保证相对位置，所以挨个遍历，找到偶数则抽出来放到最后。

```

```C++
class Solution {
public:
    void reOrderArray(vector<int> &array) {
        if(array.size() == 0)
            return;
        auto it = array.begin();
        for(int i = 0; i < array.size(); ++ i)
            if((*it & 1) == 0){
                int temp = *it;
                array.erase(it);
                array.push_back(temp);
            }
            else
                ++ it;
    }
};


```

```python
class Solution:
    def reOrderArray(self, array):
        # write code here
        length = len(array)
        pos = -1
        for i in range(length):
            if array[i] & 1:
                pos += 1
                array.insert(pos, array.pop(i))
        return array


```
