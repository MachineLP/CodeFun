```
Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right which minimizes the sum of all numbers along its path.

Note: You can only move either down or right at any point in time.

Example:

Input:
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
Output: 7
Explanation: Because the path 1→3→1→1→1 minimizes the sum.
```

### 解决问题的思路很重要，直接影响效率。   我想到的是递归， 别人用的是遍历

```python
class Solution(object):
    
    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        if not grid or grid[0] == []: return 0 
        R, C = len(grid), len(grid[0])
        for i in range(R):
            for j in range(C):
                if i != 0 or j != 0:
                    grid[i][j] += min(grid[i-1][j] if i > 0 else float("Inf"),grid[i][j-1] if j > 0 else float("Inf"))
        return grid[-1][-1]
    
    '''
    def __init__(self):
        self.min_val = float("Inf")
    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        m = len( grid )
        n = len( grid[0] )
        sum_val = 0
        self.direction(0, 0, m, n, grid, sum_val)
        return self.min_val
        
    
    def direction(self, i, j, m, n, grid, sum_val):
        
        sum_val = sum_val + grid[i][j]
        if sum_val >= self.min_val:
            return 
        
        if i==m-1 and j==n-1:
            if sum_val < self.min_val:
                self.min_val = sum_val
        
        if i < m-1 :
            self.direction(i+1, j, m, n, grid, sum_val)
        if j < n-1:
            self.direction(i, j+1, m, n, grid, sum_val)
    '''
```
