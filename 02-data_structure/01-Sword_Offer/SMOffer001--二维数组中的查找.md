```
题目描述

在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

解题思路：

首先选取数组中右上角的数字，如果该数字大于target，则剔除该数字所在的列，若小于则剔除该数字所在的行，等于的话则直接返回，相当于一直在以右上角的数字为界限，不断缩小查找范围。另外我们也可以以左下角为界限，不断缩小查找范围。

```

```cpp
//右上角
class Solution {
public:
    bool Find(int target, vector<vector<int> > array) {
        int rows = array.size(), cols = array[0].size();
        if(rows > 0 && cols > 0){
            int row = 0, col = cols - 1;
            while(row < rows && col >= 0){
                if(array[row][col] == target)
                    return true;
                else if(array[row][col] > target)
                    -- col;
                else
                    ++ row;
            }
        }
        return false;
    }
};
 
//左下角
class Solution {
public:
    bool Find(int target, vector<vector<int> > array) {
        int rows = array.size(), cols = array[0].size();
        if(rows > 0 && cols > 0){
            int row = rows - 1, col = 0;
            while(row >= 0 && col < cols){
                if(array[row][col] == target)
                    return true;
                else if(array[row][col] > target)
                    -- row;
                else
                    ++ col;
            }
        }
        return false;
    }
};


```

```python
#右上角
class Solution:
    # array 二维列表
    def Find(self, target, array):
        # write code here
        rows = len(array)
        cols = len(array[0])
        if rows > 0 and cols > 0:
            row = 0
            col = cols - 1
            while row < rows and col >= 0:
                if target == array[row][col]:
                    return True
                elif target < array[row][col]:
                    col -= 1
                else:
                    row += 1
        return False
 
#左下角
class Solution:
    # array 二维列表
    def Find(self, target, array):
        # write code here
        rows = len(array)
        cols = len(array[0])
        if rows > 0 and cols > 0:
            row = rows - 1
            col = 0
            while row >= 0 and col < cols:
                if target == array[row][col]:
                    return True
                elif target < array[row][col]:
                    row -= 1
                else:
                    col += 1
        return False



```
