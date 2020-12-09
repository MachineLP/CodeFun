```
题目描述

请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，向右，向上，向下移动一个格子。如果一条路径经过了矩阵中的某一个格子，则之后不能再次进入这个格子。 例如 a b c e s f c s a d e e 这样的3 X 4 矩阵中包含一条字符串"bcced"的路径，但是矩阵中不包含"abcb"路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入该格子。

解题思路

回溯法，在矩阵中选一个格子作为起点，该位置对应字符串的第一个字符，在在该格子周围中找与第二个字符相同的格子，由此类推，知道路径中所有字符都在矩阵中找到对应位置。

```


```C++
class Solution {
public:
    bool hasPath(char* matrix, int rows, int cols, char* str)
    {
        if(matrix == nullptr || rows < 1 || cols < 1 || str == nullptr)
            return false;
        bool *visit = new bool[rows * cols];
        memset(visit, 0, rows*cols);
        int pathLength = 0;
        for(int i = 0; i < rows; ++ i)
            for(int j = 0; j < cols; ++ j){
                if(hasPathCore(matrix, rows, cols, i, j, str, pathLength, visit)){
                    delete[] visit;
                    return true;
                }
            }
        delete[] visit;
        return false;
    }
private:
    bool hasPathCore(const char *matrix, int rows, int cols, int row, int col, const char* str, int &pathLength, bool *visit){
        if(str[pathLength] == '\0')
            return true;
        bool hasPath = false;
        if(row >= 0 && row < rows && col >= 0 && col < cols && matrix[row * cols + col] == str[pathLength] && visit[row * cols + col] == false){
            ++ pathLength;
            visit[row * cols + col] = true;
            hasPath = hasPathCore(matrix, rows, cols, row, col - 1, str, pathLength, visit) 
            || hasPathCore(matrix, rows, cols, row - 1, col, str, pathLength, visit) 
            || hasPathCore(matrix, rows, cols, row + 1, col, str, pathLength, visit) 
            || hasPathCore(matrix, rows, cols, row, col + 1, str, pathLength, visit);
            if(!hasPath){
                -- pathLength;
                visit[row * cols + col] = false;
            }
        }
        return hasPath;
    }
};


```


```python
class Solution:
    def __init__(self):
        self.visit = []
        self.flag = False
    def hasPath(self, matrix, rows, cols, path):
        # write code here
        if rows == 0 or cols == 0 or len(path) == 0:
            return self.flag
        for i in range(rows):
            self.visit.append([False]*cols)
        for i in range(rows):
            for j in range(cols):
                self.hasPathCore(matrix, i, j, rows, cols, path, 0)
                if self.flag == True:
                    return self.flag
        return self.flag
    def hasPathCore(self, matrix, i, j, rows, cols, path, cur):
        if self.flag or cur == len(path):
            self.flag = True
            return
        if i < 0 or j < 0 or i == rows or j == cols or self.visit[i][j] == True or matrix[i*cols + j] != path[cur]:
            return
        self.visit[i][j] = True
        self.hasPathCore(matrix, i+1, j, rows, cols, path, cur+1)
        self.hasPathCore(matrix, i-1, j, rows, cols, path, cur+1)
        self.hasPathCore(matrix, i, j+1, rows, cols, path, cur+1)
        self.hasPathCore(matrix, i, j-1, rows, cols, path, cur+1)
        self.visit[i][j] = False


```
