
```
The count-and-say sequence is the sequence of integers with the first five terms as following:

1.     1
2.     11
3.     21
4.     1211
5.     111221
1 is read off as "one 1" or 11.
11 is read off as "two 1s" or 21.
21 is read off as "one 2, then one 1" or 1211.

Given an integer n where 1 ≤ n ≤ 30, generate the nth term of the count-and-say sequence.

Note: Each term of the sequence of integers will be represented as a string.

 

Example 1:

Input: 1
Output: "1"
Example 2:

Input: 4
Output: "1211"
```


```python
class Solution(object):
    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        if n<=0: return ''
        if n==1: return '1'
        
        v = []
        v.append(1)
        
        for i in range(2,n+1,1):
            v = self.get_next(v)
        
        s = ''
        for per_v in v:
            s += str( per_v )
        
        '''
        s = [str(per_v) for per_v in v ]
        s = ''.join( s )'''
        return s
    
    def get_next(self, v):
        cnt = 0
        val = 0
        ret = []
        for i in range( len(v) ):
            if i==0:
                val = v[i]
                cnt = 1
                continue
            if v[i]==val:
                cnt += 1
            else:
                ret.append(cnt)
                ret.append(val)
                val = v[i]
                cnt = 1
        if cnt>0 and val>0:
            ret.append(cnt)
            ret.append(val)
        
        return ret
        
```
