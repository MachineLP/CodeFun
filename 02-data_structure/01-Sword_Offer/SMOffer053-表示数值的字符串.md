```
题目描述

请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。 但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是。

解题思路

表示数值的字符串遵循模式[A][.[B]][e|EC]，A为数值的整数部分，B为数值的小数部分，C为指数部分，同时A和B不可以同时为空。A和C均为待符号的整数，而B为无符号整数。设两个函数，scanUndignedInteger判断是否为无符号整数，scanInteger为有符号整数，再分别对A，B，C三部分进行判断。

```


```C++
class Solution {
public:
    bool isNumeric(char* str)
    {
        if(str == nullptr)
            return false;
        bool numeric = scanInteger(&str);
        if(*str == '.'){
            ++ str;
            numeric = scanUndignedInteger(&str) || numeric;
        }
        if(*str == 'e' || *str == 'E'){
                ++ str;
                numeric = numeric && scanInteger(&str);
        }
        return numeric && *str == '\0';
    }
    bool scanInteger(char** str){
        if(**str == '+' || **str == '-')
            ++(*str);
        return scanUndignedInteger(str);
    }
    bool scanUndignedInteger(char** str){
        char *before = *str;
        while(**str != '\0' && **str >= '0' && **str <= '9')
            ++(*str);
        return *str > before;
    }
};


```


```python
class Solution:
    # s字符串
    def __init__(self):
        self.flag = True
    def isNumeric(self, s):
        # write code here
        if len(s) == 0:
            return False
        if s[0] != '.':
            cur = self.scanInteger(s, 0)
        else:
            cur = 0
        if cur < len(s) and s[cur] == '.':
            cur += 1
            flag = self.flag
            self.flag = True
            cur = self.scanUnsignedInteger(s, cur)
            self.flag |= flag
        if cur < len(s) and (s[cur] == 'e' or s[cur] == 'E'):
            cur += 1
            cur = self.scanInteger(s, cur)
        return self.flag and cur == len(s)
    def scanInteger(self, s, cur):
        if cur == len(s):
            self.flag = False
        elif s[cur] == '+' or s[cur] == '-':
            cur += 1
        tempcur = self.scanUnsignedInteger(s, cur)
        return tempcur
    def scanUnsignedInteger(self, s, cur):
        temp = cur
        while cur < len(s) and s[cur] >= '0' and s[cur] <= '9':
            cur += 1
        if cur == temp:
            self.flag = False
        return cur


```
