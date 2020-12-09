```
题目描述：

请实现一个函数，将一个字符串中的每个空格替换成“%20”。例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。

解题思路：

先遍历一遍字符串，统计空格数，由此可以计算出替换空格之后的字符串的总长度，最后从字符串的后面开始复制和替换。

```

```C++
class Solution {
public:
    void replaceSpace(char *str,int length) {
        if(str == nullptr || length <= 0)
            return;
        int originalLength = strlen(str), cnt = 0;
        for(int i = 0; i < originalLength; ++ i)
            if(str[i] == ' ')
                ++ cnt;
        int newLength = originalLength + cnt * 2;
        if(newLength > length)
            return;
        int indexNew = newLength, indexOriginal = originalLength;
        for(; indexNew > indexOriginal && indexOriginal >= 0;){
            if(str[indexOriginal] == ' '){
                str[indexNew--] = '0';
                str[indexNew--] = '2';
                str[indexNew--] = '%';
            }
            else
                str[indexNew--] = str[indexOriginal];
            -- indexOriginal;
        }
    }
};

```

```python
#直接调用库函数
#replace(old, new, num),num表示最多替换几次
class Solution:
    # s 源字符串
    def replaceSpace(self, s):
        # write code here
        return s.replace(' ', '%20')
 
#手动实现
class Solution:
    # s 源字符串
    def replaceSpace(self, s):
        # write code here
        OriginalLen = len(s)
        cnt = 0
        for i in range(OriginalLen):
            if s[i] == ' ':
                cnt += 1
        s += ' ' * (cnt * 2)
        OriginalIndex = OriginalLen - 1
        CurIndex = len(s) - 1
        while OriginalIndex >= 0 and CurIndex > OriginalIndex:
            if s[OriginalIndex] == ' ':
                s = s[0:CurIndex-2] + '%20' + s[CurIndex+1:]
                CurIndex -= 3
            else:
                s = s[0:CurIndex] + s[OriginalIndex] + s[CurIndex+1:]
                CurIndex -= 1
            OriginalIndex -= 1
        return s

```
