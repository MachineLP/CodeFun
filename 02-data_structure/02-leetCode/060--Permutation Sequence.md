```
The set [1,2,3,...,n] contains a total of n! unique permutations.

By listing and labeling all of the permutations in order, we get the following sequence for n = 3:

"123"
"132"
"213"
"231"
"312"
"321"
Given n and k, return the kth permutation sequence.

Note:

Given n will be between 1 and 9 inclusive.
Given k will be between 1 and n! inclusive.
Example 1:

Input: n = 3, k = 3
Output: "213"
Example 2:

Input: n = 4, k = 9
Output: "2314"
```

```python
class Solution(object):
    def getPermutation(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: str
        """
        # 这个必须得参考
        k -= 1
        nums = list(range(1, n+1))

        places = [1]
        for i in range(1, n):
            places.append(places[i-1]*i)
        places.reverse()

        for i in range(n-1):
            j = k // places[i]
            nums.insert(i, nums.pop(i+j))
            k %= places[i]

        return ''.join([str(r) for r in nums])
```
