```
题目描述

LL今天心情特别好,因为他去买了一副扑克牌,发现里面居然有2个大王,2个小王(一副牌原本是54张^_^)...他随机从中抽出了5张牌,想测测自己的手气,看看能不能抽到顺子,如果抽到的话,他决定去买体育彩票,嘿嘿！！“红心A,黑桃3,小王,大王,方片5”,“Oh My God!”不是顺子.....LL不高兴了,他想了想,决定大\小 王可以看成任何数字,并且A看作1,J为11,Q为12,K为13。上面的5张牌就可以变成“1,2,3,4,5”(大小王分别看作2和4),“So Lucky!”。LL决定去买体育彩票啦。 现在,要求你使用这幅牌模拟上面的过程,然后告诉我们LL的运气如何， 如果牌能组成顺子就输出true，否则就输出false。为了方便起见,你可以认为大小王是0。

解题思路

直观来看，将数组排序，如果排序后的数组不是连续的，那么只要有足够的0可以补满数组中的空缺，整个数组即为连续的。接下来就简单了，首先排序，其次统计数组中相邻数字的空缺总数，如果空缺的总数小于或等于0的个数，则数组连续。另外如果碰到连续的两个数字（非0）相等，则一定不为顺子。

```


```C++
class Solution {
public:
    bool IsContinuous( vector<int> numbers ) {
        if(numbers.size() == 0)
            return false;
        sort(numbers.begin(), numbers.end());
        int cntZero = 0, cntLoss = 0, last = -1;
        for(int i = 0; i < numbers.size(); ++ i){
            if(numbers[i] == 0)
                ++ cntZero;
            else if(last == numbers[i])
                return false;
            else if(last == -1)
                last = numbers[i];
            else{
                cntLoss += numbers[i] - last - 1;
                last = numbers[i];
            }
        }
        return cntZero >= cntLoss;
    }
};


```


```python
class Solution:
    def IsContinuous(self, numbers):
        # write code here
        if len(numbers) == 0:
            return False
        cntZero, cntLoss, last = 0, 0, -1
        numbers = sorted(numbers)
        for i in range(len(numbers)):
            if numbers[i] == 0:
                cntZero += 1
            elif last == numbers[i]:
                return False
            elif last == -1:
                last = numbers[i]
            else:
                cntLoss += numbers[i] - last - 1
                last = numbers[i]
        return cntZero >= cntLoss


```
