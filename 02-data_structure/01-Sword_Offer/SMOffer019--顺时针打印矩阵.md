```
题目描述

输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，例如，如果输入如下4 X 4矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.

解题思路

这道题主要考察循环，将打印矩阵的过程线分解为打印一个个的圈，用四个循环分别打印每个圈有四条边。
```

```C++
class Solution {
public:
    vector<int>ans;
    int m, n;
    vector<int> printMatrix(vector<vector<int> > matrix) {
        ans.clear();
        if(matrix.size() == 0 || matrix[0].size() == 0)
            return ans;
        int start = 0;
        m = matrix.size();
        n = matrix[0].size();
        while((start << 1) < n && (start << 1) < m){
            printMatrixCircle(matrix, start);
            ++ start;
        }
        return ans;
    }
    void printMatrixCircle(vector<vector<int>>&matrix, int start){
        int endX = n - 1 - start, endY = m - 1 - start;
        for(int i = start; i <= endX; ++ i)
            ans.push_back(matrix[start][i]);
        if(start < endY){
            for(int i = start + 1; i <= endY; ++ i)
                ans.push_back(matrix[i][endX]);
        }
        if(start < endX && start < endY)
            for(int i = endX - 1; i >= start; -- i)
                ans.push_back(matrix[endY][i]);
        if(start < endX && start < endY - 1)
            for(int i = endY - 1; i >= start + 1; -- i)
                ans.push_back(matrix[i][start]);
    }
};


```

```python
# -*- coding:utf-8 -*-
class Solution:
    # matrix类型为二维列表，需要返回列表
    def __init__(self):
        self.ans = []
    def printMatrix(self, matrix):
        # write code here
        m, n = len(matrix), len(matrix[0])
        d = (min(m, n) +1) / 2
        for i in range(d):
            self.PrintMatrixInCircle(matrix, m, n, i)
        return self.ans
    def PrintMatrixInCircle(self, matrix, m, n, start):
        endX = m - start
        endY = n - start
        for i in range(start, endY):
            self.ans.append(matrix[start][i])
        for i in range(start+1, endX):
            self.ans.append(matrix[i][endY-1])
        if start+1 < endX:
            for i in range(endY-2, start-1, -1):
                self.ans.append(matrix[endX-1][i])
        if start + 1 < endY:
            for i in range(endX-2, start, -1):
                self.ans.append(matrix[i][start])


```
