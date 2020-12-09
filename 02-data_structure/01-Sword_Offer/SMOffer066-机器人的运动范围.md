```
题目描述

地上有一个m行和n列的方格。一个机器人从坐标0,0的格子开始移动，每一次只能向左，右，上，下四个方向移动一格，但是不能进入行坐标和列坐标的数位之和大于k的格子。 例如，当k为18时，机器人能够进入方格（35,37），因为3+5+3+7 = 18。但是，它不能进入方格（35,38），因为3+5+3+8 = 19。请问该机器人能够达到多少个格子？

解题思路

回溯法，与上一题类似，只是判断条件稍微麻烦一点，进入每个格子后，判断该格子是否可以走，可以的话，再遍历它周围的格子。

```


```C++
class Solution {
public:
    int movingCount(int threshold, int rows, int cols)
    {
        if(threshold < 0 || rows <= 0 || cols <= 0)
            return 0;
        bool *visit = new bool[rows * cols];
        memset(visit, 0, rows * cols);
        int count = movingCountCore(threshold, rows, cols, 0, 0, visit);
        delete[] visit;
        return count;
    }
private:
    int movingCountCore(int threshold, int rows, int cols, int row, int col, bool *visit){
        int count = 0;
        if(check(threshold, rows, cols, row, col, visit)){
            visit[row * cols + col] = true;
            count = 1 + movingCountCore(threshold, rows, cols, row - 1, col, visit)
            + movingCountCore(threshold, rows, cols, row + 1, col, visit)
                + movingCountCore(threshold, rows, cols, row, col - 1, visit)
                + movingCountCore(threshold, rows, cols, row, col + 1, visit);
        }
        return count;
    }
    bool check(int threshold, int rows, int cols, int row, int col, bool* visit){
        if(row >= 0 && row < rows && col >= 0 && col < cols && !visit[row * cols + col])
            if(getDigitSum(row) + getDigitSum(col) <= threshold)
                return true;
        return false;
    }
    int getDigitSum(int number){
        int sum = 0;
        while(number > 0){
            sum += number % 10;
            number /= 10;
        }
        return sum;
    }
};


```


```python
class Solution:
    def __init__(self):
        self.visit = []
        self.cnt = 0
    def movingCount(self, threshold, rows, cols):
        # write code here
        if threshold < 0 or rows <= 0 or cols <= 0:
            return 0
        for i in range(rows):
            self.visit.append([False] * cols)
        self.movingCountCore(threshold, rows, cols, 0, 0)
        return self.cnt
    def movingCountCore(self, threshold, rows, cols, i, j):
        if self.check(threshold, rows, cols, i, j):
            self.visit[i][j] = True
            self.cnt += 1
            self.movingCountCore(threshold, rows, cols, i - 1, j)
            self.movingCountCore(threshold, rows, cols, i + 1, j)
            self.movingCountCore(threshold, rows, cols, i, j - 1)
            self.movingCountCore(threshold, rows, cols, i, j + 1)
    def check(self, threshold, rows, cols, i, j):
        if i < 0 or j < 0 or i == rows or j == cols or self.visit[i][j] == True:
            return False
        return True if (self.getDigitSum(i) + self.getDigitSum(j) <= threshold) else False
    def getDigitSum(self, num):
        sum = 0
        while num:
            sum += num % 10
            num /= 10
        return sum


```
