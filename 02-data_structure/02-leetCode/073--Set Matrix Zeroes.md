```
Given a m x n matrix, if an element is 0, set its entire row and column to 0. Do it in-place.

Example 1:

Input: 
[
  [1,1,1],
  [1,0,1],
  [1,1,1]
]
Output: 
[
  [1,0,1],
  [0,0,0],
  [1,0,1]
]
Example 2:

Input: 
[
  [0,1,2,0],
  [3,4,5,2],
  [1,3,1,5]
]
Output: 
[
  [0,0,0,0],
  [0,4,5,0],
  [0,3,1,0]
]
Follow up:

A straight forward solution using O(mn) space is probably a bad idea.
A simple improvement uses O(m + n) space, but still not the best solution.
Could you devise a constant space solution?
```


```python
class Solution(object):
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        m = len(matrix[0])
        n = len(matrix)
        if m == 0 or n == 0:return
        row_col = []
        for i in range(n):
            for j in range(m):
                if matrix[i][j] == 0:
                    row_col.append([i, j])
        length = len(row_col)
        for i in range(length):
            r = row_col[i][0]
            c = row_col[i][1]
            for k in range(m):
                matrix[r][k] = 0
            for l in range(n):
                matrix[l][c] = 0
        return matrix
    
    
    '''
        if len(matrix)!=0:
            tag=0
            tag1=0
            #initialize the 0 low and 0 colunm
            for i in range(len(matrix)):
                for j in range(len(matrix[0])):
                    if matrix[i][j]==0:
                        if i==0:
                            tag=1
                        if j==0:
                            tag1=1
                        matrix[0][j]=0
                        matrix[i][0]=0
                    
            #fill 0 in the matrix (except the 0 low and 0 colunm)
            for i in range(1,len(matrix)):
                if matrix[i][0]==0:
                    for j in range(len(matrix[0])):
                        matrix[i][j]=0
            for j in range(1,len(matrix[0])):
                if matrix[0][j]==0:
                    for i in range(len(matrix)):
                        matrix[i][j]=0
            #The 0 low and 0 colunm
            if tag==1:
                for j in range(len(matrix[0])):
                    matrix[0][j] = 0
            if tag1==1:
                for i in range(len(matrix)):
                    matrix[i][0] = 0
            return matrix
    '''
```
