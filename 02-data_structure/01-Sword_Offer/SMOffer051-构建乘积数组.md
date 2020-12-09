```
题目描述

给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。不能使用除法。

解题思路

将B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]看成A[0]*A[1]*...*A[i-1]和A[i+1]*...*A[n-1]两部分的乘积，不妨设C[i] = A[0]*A[1]*...*A[i-1], D[i] = A[i+1]*...*A[n-1]，则C[i] = C[i-1]*A[i-1]，D[i] = D[i+1] * A[i+1]

```


```C++
class Solution {
public:
    vector<int> multiply(const vector<int>& A) {
        int len = A.size();
        vector<int>B(len);
        if(len == 0)
            return B;
        B[0] = 1;
        for(int i = 1; i < len; ++ i)
            B[i] = B[i-1] * A[i-1];
        double temp = 1;
        for(int i = len -2; i >= 0; -- i){
            temp *= A[i+1];
            B[i] *= temp;
        }
        return B;
    }
};


```

```python
class Solution:
    def multiply(self, A):
        # write code here
        B = []
        if len(A) == 0:
            return B
        B.append(1)
        for i in range(1,len(A)):
            B.append(A[i-1] * B[i-1])
        temp = 1
        for i in range(len(A)-2, -1, -1):
            temp *= A[i+1]
            B[i] *= temp
        return B

```
