```
C++小顶堆求Topk

求数组中的Topk数字，比如【1、4、6、7、2、9、8、3、5、0】的Top4是【6、7、8、9】。 
用小顶堆来实现，

首先用前4个元素新建一个大小为4的小顶堆，堆顶始终保存堆中的最小值。数组中的剩余数字是【2、9、8、3、5、0】
然后逐个将剩余数字与堆顶比较，如果大于堆顶，则与堆顶交换，并向下调整堆。
最后堆中保存的就是最大的4个数字。

```

```C++
#include <iostream> 
#include <vector>

using namespace std;

class MinHeap {

private: 
    int maxsize;                // 堆的大小
    void filterDown(int begin); // 向下调整堆 
    vector<int> arr;

public:
    MinHeap(int k);        // 构造函数 
    void createMinHeap(int arr[]); // 创建堆 
    void insert(int val);  // 插入元素 
    int getTop();          // 获取堆顶元素 
    vector<int> getHeap(); //获取堆中的全部元素
};

MinHeap::MinHeap(int k) { 
    maxsize = k;
}


/**
 * 创建小顶堆
 */
void MinHeap::createMinHeap(int a[]) {
    for (int i = 0; i < maxsize; i++) {
        arr.push_back(a[i]);
    }

    for(int i = arr.size() / 2 - 1; i >= 0; i--) {
        filterDown(i);
    }
}


/**
 * 插入元素
 */
void MinHeap::insert(int val) {
    if (val > getTop()) {
        arr[0] = val;
        filterDown(0);
    }
}


/**
 * 向下调整
 */
void MinHeap::filterDown(int current) {
    int end = arr.size() - 1;
    int child = current * 2 + 1; // 当前节点的左孩子
    int val = arr[current];    // 保存当前节点

    while (child <= end) {
        // 选出两个孩子中的较小孩子
        if (child < end && arr[child + 1] < arr[child])
            child++;
        if (val < arr[child]) break;
        else {
            arr[current] = arr[child]; //孩子节点覆盖当前节点
            current = child;
            child = child * 2 + 1;
        }
    }
    arr[current] = val;
}


/**
 * 获取堆顶元素
 */
int MinHeap::getTop() {
    if (arr.size() != 0)
        return arr[0];
    return NULL;
}


/** 
 * 获取堆中的全部元素
 */
vector<int> MinHeap::getHeap() {
    vector<int> heap;
    for(int i = 0; i < arr.size(); i++)
        heap.push_back(arr[i]);
    return heap;
}


int main() {
    // Test case
    int arr[] = {1,4,6,7,2,9,8,3,5,0};
    int k = 4;
    MinHeap heap(4); // 创建一个大小为4的堆 
    heap.createMinHeap(arr);
    for(int i = k; i < 10; i++) {
        heap.insert(arr[i]);
    }

    cout << "最大的四个元素是" << endl;
    vector<int> v = heap.getHeap();
    for(int i = 0; i < v.size(); i++) {
        cout << v[i] << endl;
    }
    return 0;
}
```
