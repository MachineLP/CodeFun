```
题目描述

输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。

解题思路

本题直接修改排序规则即可，对两个数字m，n来讲，能拼接成mn和nm，若mn < nm，则打印出mn，反之则为nm。

```


```C++
class Solution {
public:
    string PrintMinNumber(vector<int> numbers) {
        string ans = "";
        sort(numbers.begin(), numbers.end(), [](int a, int b){return (to_string(a) + to_string(b)) < (to_string(b) + to_string(a));});
        for(int i = 0; i < numbers.size(); ++ i)
            ans += to_string(numbers[i]);
        return ans;
    }
};


```

```python
class Solution:
    def PrintMinNumber(self, numbers):
        # write code here
        nums = sorted(numbers, cmp = lambda a, b : cmp(str(a) + str(b), str(b) + str(a)))
        return ''.join(str(num) for num in nums)


```
