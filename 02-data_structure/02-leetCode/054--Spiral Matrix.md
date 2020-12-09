
```
Given a matrix of m x n elements (m rows, n columns), return all elements of the matrix in spiral order.

Example 1:

Input:
[
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]
Output: [1,2,3,6,9,8,7,4,5]
Example 2:

Input:
[
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9,10,11,12]
]
Output: [1,2,3,4,8,12,11,10,9,5,6,7]
```


```python
class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        if not matrix:
            return []
        L = 0
        R = len(matrix[0])
        T = 0
        B = len(matrix)
        
        result = []
        while L < R and T < B:
            for i in range(L, R):
                result.append(matrix[T][i])
            for i in range(T + 1, B - 1):
                result.append(matrix[i][R - 1])
            for i in reversed(range(L, R)):
                if B - 1 != T:
                    result.append(matrix[B - 1][i])
            for i in reversed(range(T + 1, B - 1)):
                if R - 1 != L:
                    result.append(matrix[i][L])
            L += 1
            R -= 1
            T += 1
            B -= 1
        return result
        
        
        '''
        if not matrix:
            return 0
        res = []
        for i in range ( len(matrix) ):
            if i%2 ==1 :
                res += matrix[i][::-1]
            else:
                res += matrix[i]
        return res
        '''
```
