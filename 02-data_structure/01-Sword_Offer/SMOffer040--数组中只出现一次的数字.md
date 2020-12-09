```
题目描述

一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字。

解题思路

对数组中所有的数字进行异或运算，数组中出现两次的数字，通过异或位运算可以去除掉。但是如果区分这两个数字就成关键。所有数字异或之后的结果应该是我们要求两个数字的异或结果。换个思路，不在异或结果中分出两个数字，而是在异或运算前就将所有数字分为两组，刚刚求得的最后的异或结果，我们可以得到两个数字不同的位，在第二次进行异或运算的时候，将数字分为该位为0还是1。

```


```C++
class Solution {
public:
    void FindNumsAppearOnce(vector<int> data,int* num1,int *num2) {
        int mask = 1, temp = 0;
        *num1 = 0; *num2 = 0;
        for(int i = 0; i < data.size(); ++ i)
            temp ^= data[i];
        while((temp & 1) == 0){
            temp >>= 1;
            mask <<= 1;
        }
        for(int i = 0; i < data.size(); ++ i){
            if((data[i] & mask) == 0)
                *num1 ^= data[i];
            else
                *num2 ^= data[i];
        }
    }
};


```

```python
class Solution:
    # 返回[a,b] 其中ab是出现一次的两个数字
    def FindNumsAppearOnce(self, array):
        # write code here
        temp, mask, num1, num2 = 0, 1, 0, 0
        for i in range(len(array)):
            temp ^= array[i]
        while temp & 1 == 0:
            temp >>= 1
            mask <<= 1
        for i in range(len(array)):
            if array[i] & mask:
                num1 ^= array[i]
            else:
                num2 ^= array[i]
        return [num1, num2]


```
