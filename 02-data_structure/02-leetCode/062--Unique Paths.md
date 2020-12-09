```
A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

How many possible unique paths are there?


Note: m and n will be at most 100.

Example 1:

Input: m = 3, n = 2
Output: 3
Explanation:
From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
1. Right -> Right -> Down
2. Right -> Down -> Right
3. Down -> Right -> Right
Example 2:

Input: m = 7, n = 3
Output: 28
```
![image](https://assets.leetcode.com/uploads/2018/10/22/robot_maze.png)
Above is a 7 x 3 grid. How many possible unique paths are there?


```python
class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        path_length = m + n - 2
        right = n - 1
        return self.factorial_dp(path_length, right)
    
    def factorial_dp(self, n, r):
        # math.factorial(n) / (math.factorial(r) * math.factorial(n - r))
        memo_n = 0
        memo_r = 0
        memo_nr = 0
        memo = 0
        for i in range(n + 1):
            if i == 0 or i == 1:
                memo = 1
            else:
                memo = i * memo
            
            if i == r:
                memo_r = memo
            if i == n:
                memo_n = memo
            if i == n - r:
                memo_nr = memo
                
        return memo_n / (memo_r * memo_nr)
    
        
    '''
        Z = [ [0 for j in range(m)] for i in range(n) ]
        
        for j in range(m):
            Z[0][j] = 1
        for i in range(n):
            Z[i][0] = 1
        
        for i in range(1, n):
            for j in range(1, m):
                Z[i][j] = Z[i-1][j] + Z[i][j-1]
       
        return Z[-1][-1]
    '''
```
