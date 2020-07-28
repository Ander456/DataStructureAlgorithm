//
//  main.cpp
//  leetcode
//
//  Created by Alex on 2019/7/31.
//  Copyright © 2019 Alex. All rights reserved.
//

#include <iostream>
#include <vector>
#include <map>
#include <stack>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <set>
#include <queue>
#include <list>

using namespace std;


//-----------仅用栈实现队列---------------------------------------
class DequeueWithStack
{
public:
    stack<int> s1;
    stack<int> s2;

    void push_back(int i)
    {
        s1.push(i);
    }

    int pop_front()
    {
        if (s1.empty() && s2.empty())
            return -1;
        while(s2.empty())
        {
            while(!s1.empty())
            {
                s2.push(s1.top());
                s1.pop();
            }
        }
        int ret = s2.top();
        s2.pop();
        return ret;
    }
};

//-----------仅用队列实现栈---------------------------------------
class StackWithQueue
{
public:
    queue<int>* q;
    queue<int>* q2;
    
    StackWithQueue()
    {
        q = new queue<int>();
        q2 = new queue<int>();
    }
    
    ~StackWithQueue()
    {
        delete q;
        delete q2;
    }
    
    void push(int i)
    {
        q->push(i);
    }
    
    int pop()
    {
        while(q->empty() && q2->empty())
            return -1;
        while (q->size() > 1) {
            q2->push(q->front());
            q->pop();
        }
        int ret = q->front();
        q->pop();
        // swap q and q2
        queue<int>* temp = q;
        q = q2;
        q2 = temp;
        return ret;
    }
};

//-----------固定大小的数组实现栈---------------------------------------
class StackWithFixArray
{
public:
    
    int size;
    int* array;
    int idx;
    
    StackWithFixArray(int size): size(size)
    {
        array = new int[size];
        idx = 0;
    }
    
    ~StackWithFixArray()
    {
        delete []array;
        array = nullptr;
    }
    
    void push(int i)
    {
        if(idx == size)
        {
            cout << "full" << endl;
            return;
        }
        array++;
        *array = i;
        idx++;
    }
    
    int pop()
    {
        if (idx == 0)
            return -1;
        int ret = *array;
        array--;
        idx--;
        return ret;
    }
};

//-----------固定大小的数组实现队列---------------------------------------
class QueueWithFixArray
{
public:
    int start = 0;
    int idx = 0;
    int size = 0;
    int array[5] = {0,0,0,0,0};
    void push(int i)
    {
        if(size == 5)
        {
            cout << "full" << endl;
            return;
        }
        array[idx] = i;
        size++;
        idx = idx == 5-1 ? 0 : idx+1; //如果不这样 那么等pop后再push array越界了
    }
    
    int pop()
    {
        if (size == 0)
            return -1;
        int ret = array[start];
        start = start == 5-1 ? 0 : start + 1;
        size--;
        return ret;
    }
};

//-----------实现一个栈且能--获取栈内最小元素O(1)复杂度要求---------------------------------------
class MinStack
{
public:
    stack<int> s1;
    stack<int> s2;
    
    void push(int i)
    {
        s1.push(i);
        if(s2.empty())
        {
            s2.push(i);
        }
        else
        {
            int v = s2.top();
            if(i < v)
            {
                s2.push(i);
            }
            else
            {
                s2.push(v);
            }
        }
    }
    
    int pop()
    {
        if(s1.empty())
            return -1;
        int ret = s1.top();
        s1.pop();
        s2.pop();
        return ret;
    }
    
    int getMin()
    {
        return s2.top();
    }
};

// 稍微说一点感觉 就是 栈 就是一个用来保存数据的 系统有系统栈比如函数调用 上下文临时数据保存 都是系统帮你用栈存着了，其实我们自己模拟写栈也是类似 栈本来就是来临时保存数据然后恢复现场用的

//-----------堆--堆排序---------------------------------------
// 明白一件事 堆 是完全二叉树（从上到下从左到右增加树节点的机构） 但是其实我们一般用数组来表现堆(因为完全二叉树的性质)
class MaxHeap
{
public:
    
    void heapSort(int arr[], int n)
    {
        //1. 堆排序先构建堆
        buildMaxHeap(arr, n);
        //2. 构造好大顶堆后 然后 每次把最大的顶和最后一个节点交换 然后 去掉交换后的最后一个节点也就是最大值 ，然后做一次heapify
        // 重复上述动作
        int i;
        for (i = n-1; i > 0; i--) {
            swap(arr, i, 0); // 0是最大值堆顶 i是当前开始也就是最后一个
            heapify(arr, 0, i); //此时长度 就是n-1了也就是i 因为swap后最大的拿掉了相当于
        }
    }
    
    void buildMaxHeap(int arr[], int len)
    {
        //给了个无序数组为输入 我们需要输出一个堆 首先肯定是按顺序构造一个完全二叉树（对应数组）然后开始构建大顶堆 这里一般有两种方式
        //1.从上往下 2.从最后的节点向上
        //我们想一下 如果 我们从上往下 当swap了一下 当前处理的节点和它的孩子，那么它的孩子需要接着处理，如果它的孩子也需要变换
        //我们处理了孩子变换后，把一个更大的swap上来以后，它就又不满足了 还得处理一次 就二次调整了。我们写代码的时候还得知道是不是要二次调整了 去写 很蠢
        //如果我们从最后一个非叶子节点（叶子节点没有孩子不需要去比较处理）去调整，调整了一次后，当前节点必然比它所有的孩子都大，然后不断网上就好了 如果理解不了就记住吧
        int last = len - 1;
        int parentIdx = (last - 1) / 2;
        int i;
        for(i = parentIdx; i >= 0; i--)
        {
            heapify(arr, i, len);
        }
    }
    
    // i 当前索引
    // len 无序数组长度
    // 这个函数的目的 将任意数组整理成堆的形状 到底是大根堆还是小根堆 看里面的判断逻辑是 要求paret大还是小
    void heapify(int arr[], int i, int len)
    {
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        int maxIdx = i;
        // 索引<长度 （肯定的要不然越界得）
        if(left < len && arr[left] > arr[maxIdx])
            maxIdx = left;
        if(right < len && arr[right] > arr[maxIdx])
            maxIdx = right;
        // 上面找到了这个节点和他俩孩子最大的索引是哪个
        if(maxIdx != i)
        {
            //不满足大根堆定义需要交换下 每个节点的值都大于或者等于它的左右子节点的值
            swap(arr, i, maxIdx);
            //交换完以后 当前节点确实比俩孩子都大了 但是子节点改变了如果它也有自己的孩子 这里需要接着调整
            //需要调整的是被换的子节点 那么传入这个子节点索引 也就是刚才得到的maxIdx
            heapify(arr, maxIdx, len);
        }
    }
    
    void swap(int arr[], int i, int j)
    {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
    
    //-----------
    
    //arr 保存堆的数组
    //size 有效数据大小 [0-size)
    //index 要调整的节点下表
    //其实这个写法和上面的 heapify 原理类似
    void shiftDown(int arr[], int size, int index)
    {
        int parent = index; //当前要处理的点当做parent
        int child = 2 * parent + 1;
        while (child < size) //数据有效没越界
        {
            if (child + 1 < size && arr[child+1] > arr[child])
            {
                //child+1<size表示右边还有一个右节点 这里判断如果右节点大于左节点的值那么最大索引是右节点的索引
                child = child + 1;
            }
            if (arr[parent] < arr[child]) {
                //如果父节点小于子那么不符合大根堆定义 要交换
                swap(arr, parent, child);
            }
            else
            {
                break; //也就是说当前节点是它和孩子节点最大的 那不用交换
            }
            // 因为上面如果更新后子树可能不满足，更新parent指向的位置，指向上一次所找到的较大的子节点的位置，同时child也更新
            parent = child;
            child = 2 * parent + 1;
        }
    }
    
    // 同shiftDown
    // 其实这个写法和上线的 buildMaxHeap 原理类似
    void shiftUp(int arr[], int size, int index)
    {
        int child = index;
        int parent = (child-1)/2;
        while (child > 0) { //child > 0  parent就不会小于0 最次等于0 okay的
            if (arr[parent] < arr[child]) {
                //调整
                swap(arr, parent, child);
            }
            else
            {
                break;
            }
            child = parent;
            parent = (child-1)/2;
        }
    }
    
    //通过上升和下沉来建队
    void CreateHeap(int arr[], int size)
    {
        int lastIdx = size-1;
        int parent = (lastIdx-1)/2;
        for (int i = parent; i>=0; i--) {
            shiftDown(arr, size, i);
        }
    }
    
    //利用堆来实现优先级队列
    // int arr[] = new int[100]; int size = 0;
    //插入 arr[size++] = val; shiftUp(arr, size, size-1); //插入后调整
    //弹出 swap(arr[0], arr[size-1]); size--; shiftDown(arr, size, 0); //弹出后调整
    
};


//-----------仅仅用递归来反转一个栈---------------------------------------
class ReverStackWithRecusive
{
public:
    //只用递归不用另一个栈来反转一个栈
    void reverse(stack<int>& stack)
    {
        if (stack.empty()) {
            return;
        }
        int i = getAndRemoveLastElement(stack); //拿到最下面那个 其实每一次递归 系统帮忙用系统栈保存了 最下面的变量
        reverse(stack);
        stack.push(i);
    }
    
    //获取并且移除栈最下面元素 其实就是先找到最下面的元素
    int getAndRemoveLastElement(stack<int>& stack)
    {
        int result = stack.top();
        stack.pop();
        if(stack.empty()) //递归都得有终止条件
        {
            return result;
        }
        else
        {
            int last = getAndRemoveLastElement(stack);
            stack.push(result);
            return last;
        }
    }
    //ps 这个就太经典了 告诉了一个道理 栈不过就是临时存数据用来恢复现场的 递归也是系统构造一个调用栈来帮你做了而已 也揭示了一个道理 所有递归都可以用非递归实现
    // 这有点像电影院后面的挨个问前面的人 问道最前面的那个人的年龄然后依次传回来
};

//-----------窗口区间内最大最小值结构---O(N)------------------------------------
class WindowMax
{
public:
    vector<int> vec = {3,5,1,4,0,7,2};
    deque<int> *q;

    int left = 0;
    int right = 0;
    
    WindowMax()
    {
        q = new deque<int>;
    }
    
    ~WindowMax()
    {
        delete q;
    }
    
    //窗口右侧移动 （同时伴随着数字进入队列）
    void addNumToRight()
    {
        if (right == vec.size()) {
            return; // 窗口右侧已经到头了不能滑动了
        }
        while (!q->empty() && vec[q->back()] <= vec[right]) {
            //规则是这样的 如果要进队列的值比队尾大那么把比将要入队的小的全部弹出 (当然了队列里只存下表）
            q->pop_back();
        }
        q->push_back(right);
        right++;
    }
    
    //窗口左侧移动 （同时伴随着有数字出队列）
    void removeNumFromLeft()
    {
        if (left < right) { //只有 left<right 才有窗口
            if (q->front() == left)
            {
                //如果说我们队列最前面存着的下表等于我们当前的下表 说明过期了改出队列了
                q->pop_front();
            }
        }
        left++;
    }
    
    //因为维护了一个 deque 而且入队列的规则是 把将要入队小的全部出 那么肯定保证 这个操作是O(1) 而不用O(N）
    int getMax()
    {
        if (!q->empty()) {
            return q->front();
        }
        return -1;
    }
    
    //arr是一个数组
    //winSize窗口大小
    //返回值表示窗口从左到右滑动每次滑动当前窗口内的最大值
    vector<int> getMaxNumInWindow(vector<int>arr, int winSize)
    {
        vector<int> ret;
        if (arr.size() < winSize || winSize < 1) {
            return ret;
        }
        deque<int> maxQ;
        for (int i = 0; i < arr.size(); i++) //窗口固定winsize 每次移动一下窗口都会有一个数进入一个数过期出去
        {
            while (!maxQ.empty() && arr[maxQ.back()] < arr[i]) {
                maxQ.pop_back();
            }
            maxQ.push_back(i);
            if (maxQ.front() == i - winSize) {
                //表示过期了 i索引-窗口大小 肯定啊
                maxQ.pop_front();
            }
            //成窗口必须得之间size>=winsize
            if (i >= winSize - 1) {
                ret.push_back(arr[maxQ.front()]);
            }
        }
        return ret;
    }
};

//-----------并查集------------------------------------
//并查集 是一种树型结构 目的 1. 查询元素p和元素q 是否属于同一组 2. 合并元素p和元素q所在的组
class UF
{
public:
    //初始情况下 每一个元素都在一个独立的分组中 所以初始情况下 并查集中数据默认分为n个组
    //初始化数组eleAndGroup;
    //把eleAndGroup数组的索引看做每个节点存储的元素，把每个索引处的值看做是该节点所在的分组
    UF(int n) {
        count = n;
        eleAndGroup.resize(n);
        for (int i = 0; i < eleAndGroup.size(); i++) {
            eleAndGroup[i] = i;
        }
    };
    
    //获取当前并查集中数据有多少个分组
    int groupCnt()
    {
        return count;
    }
    
    //查找元素p所在的分组标识
    int find(int p)
    {
        return eleAndGroup[p];
    }
    
    //判断元素p和元素q是否在同一分组中
    bool connected(int p,  int q)
    {
        return find(p) == find(q);
    }
    
    //合并元素p和元素q的分组
    void tounion(int p, int q)
    {
        if (connected(p, q)) {
            return;
        }
        int pGroup = find(p);
        int qGroup = find(q);
        for (int i = 0; i < eleAndGroup.size(); i++) {
            if (eleAndGroup[i] == pGroup) {
                eleAndGroup[i] = qGroup;
            }
        }
        count--;
    }
private:
    vector<int> eleAndGroup; //记录节点元素和该元素所在的分组标识
    int count; // 记录并查集中数据的分组的个数
};

class UFTree
{
public:
    UFTree(int n) {
        count = n;
        eleAndGroup.resize(n);
        for (int i = 0; i < eleAndGroup.size(); i++) {
            eleAndGroup[i] = i;
        }
    };
    
    //获取当前并查集中数据有多少个分组
    int groupCnt()
    {
        return count;
    }
    
    int find(int p)
    {
        while (true) {
            if (p == eleAndGroup[p]) { //因为p这个值 在eleAndGroup是作为索引的 索引处的值是它的父节点
                return p;
            }
            p = eleAndGroup[p];
        }
    }
    
    //判断元素p和元素q是否在同一分组中
    bool connected(int p,  int q)
    {
        return find(p) == find(q);
    }
    
    //合并元素p和元素q的分组
    void tounion(int p, int q)
    {
        //先找到p所在分组的根节点 然后找q所在分组的根节点
        int pRoot = find(p);
        int qRoot = find(q);
        if (pRoot == qRoot) {
            return;
        }
        //否则让p所在树的根节点的父节点 为q所在树的根节点即可
        eleAndGroup[pRoot] = qRoot; //如果就这样简单的取巧设置可以实现 但是 可能会造成 线性的合并 导致 find 函数最坏时间复杂度是O(n)
        count--;
    }
private:
    vector<int> eleAndGroup; //里面每一个索引都任然像UF表示当前存着的元素 而索引处的值 改存为当前元素的父节点
    int count; // 记录并查集中数据的分组的个数
};

class UF_Tree_Weighted
{
public:
    UF_Tree_Weighted(int n) {
        count = n;
        eleAndGroup.resize(n);
        for (int i = 0; i < eleAndGroup.size(); i++) {
            eleAndGroup[i] = i;
        }
        
        sz.resize(n);
        for (int i = 0; i < sz.size(); i++) {
           sz[i] = 1;
       }
    };
    
    //获取当前并查集中数据有多少个分组
    int groupCnt()
    {
        return count;
    }
    
    int find(int p)
    {
        while (true) {
            if (p == eleAndGroup[p]) { //因为p这个值 在eleAndGroup是作为索引的 索引处的值是它的父节点
                return p;
            }
            p = eleAndGroup[p];
        }
    }
    
    //判断元素p和元素q是否在同一分组中
    bool connected(int p,  int q)
    {
        return find(p) == find(q);
    }
    
    void tounion(int p, int q)
    {
        //先找到p所在分组的根节点 然后找q所在分组的根节点
        int pRoot = find(p);
        int qRoot = find(q);
        if (pRoot == qRoot) {
            return;
        }
        //判断proot对应的树深 还是qroot对应的树深 把小树合并到大树
        if (sz[pRoot] < sz[qRoot]) {
            eleAndGroup[pRoot] = qRoot;
            sz[qRoot] += sz[pRoot];
        }
        else
        {
            eleAndGroup[qRoot] = pRoot;
            sz[pRoot] += sz[qRoot];
        }
        count--;
    }
    
private:
    vector<int> eleAndGroup;
    int count; // 记录并查集中数据的分组的个数
    vector<int> sz; //记录所在树的节点数
};

//-----------线段树------------------------------------
//通常用来求 左侧或者右侧 大于小于当前元素的个数
struct SegmentTreeNode
{
    int start;
    int end;
    int count;
    SegmentTreeNode* left;
    SegmentTreeNode* right;
    SegmentTreeNode(int _start, int _end):start(_start),end(_end)
    {
        left = nullptr;
        right = nullptr;
        count = 0;
    }
};

class SegmentTree
{
public:
    // 给定起点和终止范围 构造线段树
    SegmentTreeNode* build(int start, int end)
    {
        if (start > end) {
            return nullptr;
        }
        SegmentTreeNode* root = new SegmentTreeNode(start, end);
        if (start == end)
        {
            root->count = 0;
        }
        else
        {
            int mid = start + (end - start)/2;
            root->left = build(start, mid);
            root->right = build(mid+1, end);
        }
        return root;
    }
    
    //给定起点和终止求着个范围内有多少个节点元素
    int count(SegmentTreeNode* root, int start, int end)
    {
        if (root == nullptr) {
            return 0;
        }
        if (start == root->start && end == root->end) {
            return root->count;
        }
        //二分 然后 加起来
        int mid = root->start + (root->end - root->start)/2;
        int leftcount = 0, rightcount = 0;
        
        if (start <= mid) { //左半部分
            if (mid < end)
                leftcount = count(root->left, start, mid);
            else
                leftcount = count(root->left, start, end); //如果中点不仅包含左边还包含右边
        }
        
        if (mid < end) { //右半部分
            if (start <= mid)
                rightcount = count(root->right, mid+1, end);
            else
                rightcount = count(root->right, start, end);
        }
        return leftcount + rightcount;
    }
    
    // 往线段树中插入
    void insert(SegmentTreeNode* root, int index, int val)
    {
        if (root->start == index && root->end == index) { // 到了要插入的目标位置
            root->count += val;
            return;
        }
        // 二分查找要插入的位置
        int mid = root->start + (root->end - root->start)/2;
        if (index >= root->start && index <= mid) {
            insert(root->left, index, val);
        }
        if (index <= root->end && index > mid) {
            insert(root->right, index, val);
        }
        root->count = root->left->count + root->right->count;
    }
};

//-------------------------字典树----------------------------
//根节点什么都不包含 插入一个字符就给根节点加一个节点 再插入 如果里面child有那么接着往下走
//通常用来解决字符串的搜索问题
struct TrieNode
{
    bool isEnd;
    TrieNode* next[26] = {nullptr};
    TrieNode(){}
    ~TrieNode()
    {
        for (int i = 0; i < 26; i++) {
            if (next[i]) {
                delete next[i];
            }
        }
    }
};
class Trie
{
public:
    TrieNode* root;
    Trie()
    {
        root = new TrieNode();
    }
    ~Trie()
    {
        delete root;
    }
    void insert(const string& s)
    {
        TrieNode* cur = root;
        for (int i = s.length()-1; i >= 0; i--) {
            int t = s[i] - 'a';
            if (cur->next[t] == nullptr) {
                cur->next[t] = new TrieNode();
            }
            cur = cur->next[t];
        }
        cur->isEnd = true;
    }
    bool search(const string& word)
    {
        TrieNode* cur = root;
        for (auto c : word)
        {
            int t = c - 'a';
            if (cur->next[t] == nullptr) {
                return false;
            }
            cur = cur->next[t];
        }
        if (cur->isEnd) {
            return true;
        }
        return false;
    }
};

//-----------图------------------------------------
//根据图的定义 我们只需要 表示清楚 点和边就能表达一个图
// 1. 图中所有的点
// 2. 图中所有的顶点的边
// 图的两种存储结构 邻接矩阵 邻接表
// 邻接矩阵：也就是二维数组 我们把数组的索引代表顶点， 索引处存的值表示为两个顶点是否相连 1相连 0不相连
// 邻接矩阵的缺点 V表示顶点数那么 它需要占用的空间数是 V的2次方 因为是个二维矩阵

// 邻接表: 使用一个大小为V的数组 存储Queue[V]adj, 把索引看做是顶点 每个索引处存储了一个队列 里面存的是与该顶点相邻的其他顶点
// 相比邻接矩阵 空间复杂度降低了
class Graph
{
public:
    Graph(int v):V(v)
    {
//        cout << "Graph Constructor" << endl;
        E = 0;
        adj = new vector<list<int>*>(v);
//        adj2 = new queue<int>[10];
        for (int i = 0; i < adj->size(); i++) {
            adj->at(i) = new list<int>();
        }
//        p = new int(1);
    }
    ~Graph()
    {
//        cout << "Graph Deconstructor" << endl;
        for (int i = 0; i < adj->size(); i++) {
            adj->at(i)->clear();
            delete adj->at(i);
        }
        delete adj;
//        delete[] adj2;
//        delete p;
    }
    
    int getV()
    {
        return V;
    }
    int getE()
    {
        return E;
    }
    void addEdge(int v, int w)
    {
        //在无向图中边是没有方向的 所以该边是从v-w 又可以说是从w-v的边
        adj->at(v)->push_back(w);
        adj->at(w)->push_back(v);
        E++; // 边数量加一
    }
    
    list<int>* getAdj(int v)
    {
        return adj->at(v);
    }
private:
    //定点数
    int V;
    //边数
    int E;
    //邻接表
    vector<list<int>*>* adj;
//    queue<int>* adj2;
//    int* p;
};

//图的搜索 深度优先搜索 广度优先搜索
class DepthFirstSearch
{
public:
    DepthFirstSearch(Graph* G, int s)
    {
        markd.resize(G->getV());
        cnt = 0;
        dfs(G, s);
    }
    //深度遍历图G 找到所有和它连通的顶点
    void dfs(Graph* G, int v)
    {
        stk.push(v);
        markd[v] = true;
        while (!stk.empty()) {
            int w = stk.top();
            stk.pop();
            list<int>* q = G->getAdj(w);
            for (int e : *q)
            {
                if (!markd[e]) {
                    stk.push(e);
                    markd[e] = true;
                    cnt++;
                }
            }
        }
//        递归版本
//        markd[v] = true; //将v顶点标记为已经搜索
//        list<int>* q = G->getAdj(v);
//        for (int e : *q) //遍历候选人 也就是这次有哪些可以选择
//        {
//            if (!markd[e]) {
//                dfs(G, e);
//            }
//        }
//        cnt++; //相通着的节点数加一
    }
    bool ismakred(int v)
    {
        return markd[v];
    }
    int getCnt()
    {
        return cnt;
    }
private:
    vector<bool> markd;
    int cnt;
    stack<int> stk; //用stack就不需要递归
};
//图的搜索广度优先
class BreadFirstSearch
{
public:
    BreadFirstSearch(Graph* G, int s)
    {
        markd.resize(G->getV());
        cnt = 0;
        bfs(G, s);
    }
    void bfs(Graph* G, int v)
    {
        q.push(v);
        markd[v] = true;
        while (!q.empty()) {
            int w = q.front();
            q.pop();
            for (int n : *G->getAdj(w))
            {
                if (!markd[n]) {
                    q.push(n);
                    markd[n] = true;
                    cnt++;
                }
            }
        }
//        递归版本
//        markd[v] = true;
//        q.push(v);
//        while (!q.empty()) {
//            int w = q.front();
//            q.pop();
//            for (int n : *G->getAdj(w))
//            {
//                if (!markd[n]) {
//                    bfs(G, n);
//                }
//            }
//        }
//        cnt++;
    }
    bool ismarked(int v)
    {
        return markd[v];
    }
    int getCnt()
    {
        return cnt;
    }
private:
    int cnt;
    vector<bool> markd;
    queue<int> q; //广度优先搜索用队列
};

//----------------------------------------------------------------------------------------
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

template<typename T>
struct BSTNode
{
    BSTNode<T>* left;
    BSTNode<T>* right;
    T val;
    BSTNode(const T& value = T()): left(nullptr), right(nullptr), val(value) {}; // const T& value = T()默认参数
};

template<typename T>
class BSTree
{
public:
    typedef BSTNode<T> Node;
    typedef Node* pNode;
    
    void levelorder()
    {
        if (root) {
            // bfs 思路 一层一层找 用队列
            queue<pNode> q;
            pNode cur = root;
            q.push(cur);
            while (!q.empty()) {
                cur = q.front();
                q.pop();
                std::cout << cur->val << std::endl;
                if (cur->left) {
                    q.push(cur->left);
                }
                if (cur->right) {
                    q.push(cur->right);
                }
            }
        }
    }
    
    void inorder()
    {
        inorder(root);
    }
    
    void inorder(pNode root)
    {
        if (root) {
            inorder(root->left);
            std::cout << root->val << std::endl;
            inorder(root->right);
        }
    }
    
    void inorder2(pNode root)
    {
        if (root) {
            // 中序遍历 左 根 右 到了最下面的左每次pop 看看此时的根 有没有 右 有就push
            stack<pNode> stk;
            pNode cur = root;
            while (!stk.empty() || cur != nullptr) {
                if (cur != nullptr) {
                    stk.push(cur);
                    cur = cur->left;
                }
                else
                {
                    cur = stk.top();
                    stk.pop();
                    std::cout << cur->val << std::endl;
                    cur = cur->right;
                }
            }
        }
    }
    
    void preorder()
    {
        preorder(root);
    }
    
    void preorder(pNode root)
    {
        if (root) {
            std::cout << root->val << std::endl;
            preorder(root->left);
            preorder(root->right);
        }
    }
    
    //非递归形式的技巧就是 pop出来的时候操作
    void preorder2(pNode root)
    {
        if (root) {
            stack<pNode> stk;
            stk.push(root);
            pNode cur = root;
            while (!stk.empty()) {
                // 因为 前序遍历是 根 左 右  这里放到栈里要想先访问左 得 后推入左
                cur = stk.top();
                stk.pop();
                std::cout << cur->val << std::endl;
                if (cur->right) {
                    stk.push(cur->right);
                }
                if (cur->left) {
                    stk.push(cur->left);
                }
            }
        }
    }
    
    void postorder()
    {
        postorder(root);
    }
    
    void postorder(pNode root)
    {
        if (root) {
            postorder(root->left);
            postorder(root->right);
            std::cout << root->val << std::endl;
        }
    }
    
    void postorder2(pNode root)
    {
        if (root) {
            // 后续遍历 左 右 根
            stack<pNode> stk1;
            stack<pNode> stk2;
            pNode cur = root;
            stk1.push(cur);
            while (!stk1.empty()) {
                cur = stk1.top();
                stk1.pop();
                stk2.push(cur); // 下面是先根 后左 后右 那么stk1 pop 就是 右左 这样来进入stk2 那stk2 pop出来的就是先左 符合 后续 左右根
                if (cur->left) {
                    stk1.push(cur->left);
                }
                if (cur->right) {
                    stk1.push(cur->right);
                }
            }
            while (!stk2.empty()) {
                cur = stk2.top();
                stk2.pop();
                std::cout << cur->val << std::endl;
            }
        }
    }
    
    pNode search(const T& value)
    {
        pNode cur = root;
        while (cur) {
            if (cur->val == value)
                return cur;
            else if (cur->val > value)
                cur = cur->left;
            else
                cur = cur->right;
        }
        return cur;
    }
    
    bool insert(const T& value)
    {
        if (root == nullptr) {
            root = new Node(value);
            return true;
        }
        
        pNode cur = root;
        pNode parent = nullptr;
        while (cur) {
            parent = cur;
            if (cur->val == value)
                return false;
            else if (cur->val > value)
                cur = cur->left;
            else
                cur = cur->right;
        }
        
        cur = new Node(value);
        
        if (parent->val > value)
            parent->left = cur;
        else
            parent->right = cur;
        
        return true;
        
    }
    
    bool del(const T& value)
    {
        if (root == nullptr)
            return false;
        
        // 分情况 1. 要删除的节点只有 一个孩子 那么直接 要被删除的节点的父亲节点指向孩子
        // 2. 要删除的节点 有两个孩子 那么找到右子树 中最小的 让它来代替要删除的节点
        pNode parent = nullptr;
        pNode target = nullptr;
        
        pNode cur = root;
        while (cur) {
            if (cur->val > value)
            {
                parent = cur;
                cur = cur->left;
            }
            else if (cur->val < value)
            {
                parent = cur;
                cur = cur->right;
            }
            else
                break;
        }
        
        if (cur == nullptr)
            return false;
        
        if (cur->left == nullptr && cur->right == nullptr)
        {
            // cur 为叶子节点直接删了它就行
            if (parent->left == cur)
                parent->left = nullptr;
            else
                parent->right = nullptr;
            delete cur;
            cur = nullptr;
        }
        else if (cur->left == nullptr) {
            // 只有右孩子了
            if (parent->left == cur)
                parent->left = cur->right;
            else
                parent->right = cur->right;
            
            delete cur;
            cur = nullptr;
        }
        else if (cur->right == nullptr)
        {
            //只有左孩子了
            if (parent->left == cur)
                parent->left = cur->left;
            else
                parent->right = cur->left;
            
            delete cur;
            cur = nullptr;
        }
        else if (cur->left != nullptr && cur->right != nullptr) {
            //两个孩子都在 找到右孩子里最小的
            pNode minNode = cur->right;
            pNode minNodeP = cur;
            while (minNode->left) {
                minNodeP = minNode;
                minNode = minNode->left;
            }
            //替换下值 删掉这个最小的
            cur->val = minNode->val;
            
            if (minNodeP->right == minNode)
                minNodeP->right = minNode->right;
            else
                minNodeP->left = minNode->left;
            
            delete minNode;
            minNode = nullptr;
        }
        return true;
    }
    
private:
    pNode root;
};

class Solution
{
public:
    bool isPalindrome(string s) {
        int start = 0;
        int end = s.size() - 1;
        while(start < end)
        {
            while(!isalnum(s[start]) && start < end)
                start++;
            
            while(!isalnum(s[end]) && start < end)
                end--;

            if(toupper(s[start]) == toupper(s[end]))
            {
                start++;
                end--;
            }
            else
            {
                return false;
            }
        }
        return true;
    }
    
	//leetcode 209
    int minSubArrayLen(int s, vector<int>& nums) {
       int n = nums.size();
       if (n == 0)
           return 0;

       int left = 0;
       int right = 0;
       
       int ans = INT_MAX;

       while (right < n)
       {
           s -= nums[right];
           while (s <= 0)
           {
               ans = min(ans, right-left+1);
               s += nums[left];
               left++;
           }
           right++;
       }
       return ans == INT_MAX ? 0 : ans;
   }

	//leetcode 剑指offer46
	int translateNum(int num)
	{
		//和爬楼梯一样 爬楼梯是每次1层或者2层 这个每次选择一个或者两个 达到最后（走完楼梯或者翻译完）有多少方法
		string s = to_string(num);
		int dp[11];//这道题为啥是一维的dp因为 这里面条件的变化维度只有s的长度
		dp[0] = 1;//终止条件 当只有0个数让我们选那就是1
		dp[1] = 1;//当只有一个选择那就是它 1
		for (int i = 1; i < s.size(); i++)
		{
			//这里就是状态转移方程了
			//dp[i]表示 s的第i个索引的时候有多少种方法  放到这个题我们想下 如果s[i-1]和s[i-2]能合成一个数（符合<26 不是0？这种）那么就加上它
			//否则就还是原来那么多方法，
			if (s[i-1] == '0' || s.substr(i-1, 2) > "25")//说白了就是能合成一个字符来翻译还是不能 因为要求多少种不同的翻译方法 如果不能合成就只能单独翻译成一个一个的那就是只有这一种方法 那么前面再有多少个不同的到这还是一样的
			{
				dp[i+1] = dp[i];//因为不和规则所以不能选 所以方法数没变
			}
			else
			{
				dp[i+1] = dp[i-1] + dp[i];
			}
		}
		return dp[s.size()];
	}

	//leetcode剑指offer28
	bool isSymmetric(TreeNode* root)
	{
		if (root == nullptr)
		{
			return true;
		}
		else
		{
			return isSame(root->left, root->right);
		}
	}
	//判断是不是对称二叉树就是左右子树对称那么肯定对称 是吧
	bool isSame(TreeNode* left, TreeNode* right)
	{
		if (left == nullptr && right == nullptr)
			return true;
		if (left == nullptr || right == nullptr || left->val != right->val)
			return false;
		return isSame(left->left, right->right) && isSame(left->right, right->left);
	}

	//leetcode 67
	string addBinary(string a, string b)
	{
		//挨个加 肯定有进位 用carry表示进位
		int carry = 0;
		int count = max(a.size(), b.size());
		reverse(a.begin(), a.end());
		reverse(b.begin(), b.end());
		string ans;
		for (int i = 0; i < count; i++)
		{
			if (i >= a.size())
				carry += 0;//空位补0
			else
				carry += (a[i] == '1' ? 1 : 0);
			if (i >= b.size())
				carry += 0;
			else
				carry += (b[i] == '1' ? 1 : 0);
			ans.push_back(carry % 2 == 1 ? '1' : '0');
			carry /= 2;
		}
		//再把ans reverse下就行
		if (carry == 1)
			ans.push_back('1');
		reverse(ans.begin(), ans.end());
		return ans;
	}
	//leetcode 66
	vector<int> plusOne(vector<int>& digits){
		reverse(digits.begin(), digits.end());
		int carry = 0;
		int val = 0;
		carry++;
		for (int i = 0; i < digits.size(); i++)
		{
			val = digits[i] + carry;
			carry = val / 10;
			digits[i] = val % 10;
		}
		if (carry == 1)
		{
			digits.push_back(1);
		}
		reverse(digits.begin(), digits.end());
		return digits;
	}
	//leetcode 1431
	vector<bool> kidsWithCandies(vector<int>& candies, int extraCandies)
	{
		vector<bool> res;
		int max_cnt = 0;
		for (auto c : candies)
		{
			max_cnt = max(c + extraCandies, max_cnt);
		}
		for (auto c : candies)
		{
			if (c + extraCandies >= max_cnt)
			{
				res.push_back(true);
			}
			else
			{
				res.push_back(false);
			}
		}
		return res;
	}
	//leetcode剑指offer29
	vector<int> spiralOrder(vector<vector<int>>& matrix) {
		vector<int> res;
		if (matrix.size() == 0)
			return res;
		int top = 0;
		int down = matrix.size()-1;
		int left = 0;
		int right = matrix[0].size()-1;
		int index = 0;
		int count = (down+1) * (right+1);
		while (index < count)
		{
			for (int i = left; i <= right; i++)
			{
				res.push_back(matrix[top][i]);
				index++;
			}
			top++;
			if (index == count)
				break;
			for (int i = top; i <= down; i++)
			{
				res.push_back(matrix[i][right]);
				index++;
			}
			right--;
			if (index == count)
				break;
			for (int i = right; i >= left; i--)
			{
				res.push_back(matrix[down][i]);
				index++;
			}
			down--;
			if (index == count)
				break;
			for (int i = down; i >= top; i--)
			{
				res.push_back(matrix[i][left]);
				index++;
			}
			left++;
			if (index == count)
				break;
		}
		return res;
	}
    
    int findKthLargest(vector<int>& nums, int k) {
        // 1. build max heap
        buildMaxHeap(nums, nums.size());
        // 2. sort
        heap_sort(nums);
        // 3. get kth larget
        return nums[nums.size() - k];
    }

    void heap_sort(vector<int>& nums)
    {
        // 每次都是顶（最大）和最后一个交换 然后cut掉最后一个
        int last = nums.size() - 1;
        for (int i = last; i > 0; i--)
        {
            swap(nums, i, 0);
            heapify(nums, 0, i);
        }
    }

    void swap(vector<int>& nums, int i, int j)
    {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    void buildMaxHeap(vector<int>& nums, int size)
    {
        int last_p = (size -1 - 1)/2;
        for (int i = last_p; i >= 0; i--)
        {
            heapify(nums, i, size);
        }
    }

    void heapify(vector<int>& nums, int i, int size)
    {
        int parent = i;
        int child1 = 2 * parent + 1;
        int child2 = 2 * parent + 2;
        int max = parent;
        if (child1 < size && nums[child1] > nums[max])
            max = child1;
        if (child2 < size && nums[child2] >= nums[max])
            max = child2;
        
        if (max != parent)
        {
            swap(nums, parent, max);
            heapify(nums, max, size);
        }
    }
    //leetcode 718
    int findLength(vector<int>& A, vector<int>& B) {
//        int ans = 0;
//
//        for (int i = 0; i < A.size(); i++)
//        {
//            for (int j = 0; j < B.size(); j++)
//            {
//                int k = 0;
//                while (A[i+k] == B[j+k])
//                {
//                    k++;
//                }
//                ans = max(ans, k);
//            }
//        }
//
//        return ans;
        // 上面BigO(n3) 时间复杂度太高了
        // 最长公共子串问题 基本就是 动态规划
        // 定义：dp[i][j]为A以i结尾，B以j结尾的最长公共子数组长度 为什么是二维dp？因为长度和两个维度因素有关系 A的长度 B的长度
        // 如果 A[i] == B[j] 那么dp[i][j] = dp[i-1][j-1] + 1
        // 如果 A[i] != B[j] 则 dp[i][j] = 0
        // 注意i j是表示以 i j 索引结尾的子数组（子串）
        int n = A.size(), m = B.size();
        // 初始化为0
        vector<vector<int>> dp (n + 1, vector<int>(m + 1, 0));
        //上面为啥要n+1 m+1 固定套路
        int ans = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                if (A[i] == B[j]) {
                    dp[i+1][j+1] = dp[i][j] + 1;
                    ans = max(ans, dp[i+1][j+1]);
                }
            }
        }
        return ans;
    }
    
    //leetcode 347
    vector<int> topKFrequent(vector<int>& nums, int k) {
        vector<int> ans;
        map<int, int> counter;
        for (auto n : nums)
        {
            counter[n]++;
        }
        
        for (auto w : counter)
        {
            cout << w.first << ":" << w.second << endl;
        }

        return ans;
    }
    struct comp {
        bool operator() (vector<int> a, vector<int> b)
        {
            return (abs(a[0]) * abs(a[0]) + abs(a[1]) * abs(a[1])) > (abs(b[0]) * abs(b[0]) + abs(b[1]) * abs(b[1]));
        }
    };

    vector<vector<int>> kClosest(vector<vector<int>>& points, int K) {
        priority_queue<vector<int>, vector<vector<int>>, comp> pq;
        for (int i = 0; i < points.size(); i++)
        {
            pq.push(points[i]);
        }
        vector<vector<int>> ans;
        for (int i = 0; i < K; i++)
        {
            ans.push_back(pq.top());
            pq.pop();
        }
        return ans;
    }
    
    int kthSmallest(vector<vector<int>>& matrix, int k) {
        priority_queue<int, vector<int>, greater<int>> pq;
        for (auto r : matrix)
        {
            for (auto i : r)
            {
                pq.push(i);
            }
        }
        while (k > 1)
        {
            pq.pop();
            k--;
        }
        return pq.top();
    }
    
    //leetcode 1028
    TreeNode* recoverFromPreorder(string S) {
        // 通过寻找 - 来判断是第几层
        stack<TreeNode* > path; // 存储树的节点 size() 就是 level + 1
        int pos = 0; // 索引字符串S的位置
        while (pos < S.size())
        {
            //每一次迭代都看下当前节点的level 深度
            int level = 0;
            // 1.查找 -
            while (S[pos] == '-') {
                level++;
                pos++;
            }
            // 2.查找 digits
            int value = 0; //数字可能连续
            while (pos < S.size() && isdigit(S[pos])) {
                value = value * 10 + (S[pos] - '0');
                pos++;
            }
            TreeNode* node = new TreeNode(value);
            // 找到了 数字 我们得确定它是第几层的 是谁的子节点 是左还是右
            if (level == path.size()) {
                //level和stack里存着的节点数相同 有两种情况 一种 都是0 另一种 当前这个应该是上一个的左
                if (!path.empty()) {
                    path.top()->left = node;
                }
            }else{
                //如果 level和里面节点不相等 就说明 是右节点
                while (level != path.size()) {
                    path.pop();
                }
                path.top()->right = node;
            }
            path.push(node); //把当前节点推入stack
        }
        while (path.size() > 1) {
            path.pop();
        }
        return path.top();
    }
    
   int lastStoneWeight(vector<int>& stones) {
       if (stones.size() == 1)
       {
           return stones[0];
       }
       priority_queue<int, vector<int>> maxHeap;
       for (auto n : stones)
       {
           maxHeap.push(n);
       }
       int val = 0;
       int count = 0;
       while (maxHeap.size() > 0)
       {
           val = maxHeap.top() == val ? 0 : abs(maxHeap.top() - val);
           maxHeap.pop();
           count++;
           if (val != 0 && maxHeap.size() == 0) {
               return val;
           }
           if (count == 2) {
               count = 0;
               if (val != 0) {
                   maxHeap.push(val);
                   val = 0;
               }
           }
       }
       if (val != 0)
           return val;

       return 0;
   }
    
    // leetcode 63
    // 动态规划 套路 1.2.3
    // 1. 状态表示： 这里申明dp[i][j] 这里dp[i][j]表示为走到第i行第j列的方法书（题目就是求方法数） 为啥是二位dp因为 这道题状态和两个维度 相关 右和下
    // 2. 状态转移计算： 怎么推导出dp[i][j] ？ 一种情况是ob[i][j] == 1表示有障碍物 那么dp[i][j] = 0 为啥？那个地方都障碍堵住了怎么去？去不了肯定是0啊
    // 另一种情况是 ob[i][j] == 0 无障碍物，那怎么去的这呢？从左变和上面（因为移动方式只有向右向下） 那就是 dp[i][j] = dp[i-1][j] + dp[i][j-1] 没了
    // 3. 初始化也就是base基本情况：第一行 第一列 因为只能向右 向下 所以 第一行和第一列就得看 如果里面有了障碍那么它和它后面的点都到不了了。
    // ps :这里容易出现思考误区：遇到障碍我要想办法绕着走。这种“动态”的思考，不符合 DP “状态” 的思路 状态就是这个当前的状态别想别的
    // 我们思考单个点的“状态”：障碍点，是无法走入的点，是抵达方式数为 0 的点，是无法提供给别的点方式数的点 无法给别的点做贡献的点
    /*
     动态规划的题目分为两大类，
        一种是求最优解类，典型问题是背包问题，
        另一种就是计数类，比如这里的统计方案数的问题，
        首先它们都存在一定的递推性质，前者的递推性质还有一个名字，叫做
       「最优子结构」——即当前问题的最优解取决于子问题的最优解，
        后者类似，当前问题的方案数取决于子问题的方案数。所以在遇到求方案数的问题时，我们可以往动态规划的方向考虑。
     */
     int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
         int m = obstacleGrid.size();
         int n = obstacleGrid[0].size();
         if (obstacleGrid[0][0] == 1) {
             return 0;
         }
         //1. 状态表示
         vector<vector<int>> dp(m, vector<int>(n));
         //2. 初始化base 首行首列
         dp[0][0] = 1; //起始已经占了无法再次到达
         for (int i = 1; i < m; i++)
         {
             dp[i][0] = obstacleGrid[i][0] || dp[i-1][0] == 0 ? 0 : 1; //这表示第一列 如果有障碍下面都是1 不可到达
         }
         for (int i = 1; i < n; i++)
         {
             dp[0][i] = obstacleGrid[0][i] || dp[0][i-1] == 0 ? 0 : 1;  //这表示第一行 如果有障碍右面都是1 不可到达
         }
         //2. 状态转移计算
         for (int i = 1; i < m; i++) {
             for (int j = 1; j < n; j++) {
                 if (obstacleGrid[i][j] == 0) {
                     dp[i][j] = dp[i-1][j] + dp[i][j-1];
                 }
                 else
                 {
                     dp[i][j] = 0;
                 }
             }
         }
         return dp[m-1][n-1];
     }
    
    vector<int> divingBoard(int shorter, int longer, int k) {
        vector<int> ans;
        vector<int> helper;
        dfs(ans, helper, k, shorter, longer);
        sort(ans.begin(), ans.end());
        return ans;
    }

    int getSum(vector<int>& c)
    {
        int s = 0;
        for (auto n : c)
            s += n;
        return s;
    }

    void dfs(vector<int>& res, vector<int>& helper, int k, int shorter, int longer)
    {
        //1. 终止条件
        if (helper.size() == k)
        {
            int sum =  getSum(helper);
            if (find(res.begin(), res.end(), sum) == res.end())
            {
                res.push_back(sum);
            }
            return;
        }

        //2. 遍历候选人+筛选条件
        //a. 短
        helper.push_back(shorter);
        dfs(res, helper, k, shorter, longer);
        helper.pop_back();
        
        //b. 长
        helper.push_back(longer);
        dfs(res, helper, k, shorter, longer);
        helper.pop_back();
    }
    
    TreeNode* reConstructBinaryTree(vector<int>& pre, vector<int>& mid)
    {
        if (pre.size() == 0) {
            return nullptr;
        }
        
        vector<int> leftPre, leftMid, rightPre, rightMid;
        TreeNode* cur = new TreeNode(pre[0]);
        int rootPos = 0;
        for (int i = 0; i < mid.size(); i++)
        {
            if (mid[i] == pre[0]) { // 从中序中找到 前序的第一个也就是 根节点
                rootPos = i;
                break;
            }
        }
        for (int i = 0; i < mid.size(); i++)
        {
            //找到了中序中的根也就知道 了 左右子树分别都是谁了
            if (i < rootPos) {
                leftMid.push_back(mid[i]);
                leftPre.push_back(pre[i + 1]);
            }
            else if (i > rootPos)
            {
                leftMid.push_back(mid[i]);
                leftPre.push_back(pre[i]);
            }
        }
        cur->left = reConstructBinaryTree(leftPre, leftMid);
        cur->right = reConstructBinaryTree(rightPre, rightMid);
        return cur;
    }
    
    TreeNode* reConstructBinaryTree1(vector<int>& back, vector<int>& mid)
    {
        if (back.size() == 0) {
            return nullptr;
        }
        
        //目的就是 把back最后一个也就是根 在mid中找到 那样 就可以分开 左右子树了 然后分别 递归
        TreeNode* cur = new TreeNode(back[back.size()-1]);
        int rootPos = 0;
        for (int i = 0 ; i < mid.size(); i++) {
            if (mid[i] == back[back.size()-1]) {
                rootPos = i;
                break;
            }
        }
        vector<int> leftBack, leftMid, rightBack, rightMid;
        for (int i = 0; i < mid.size(); i++) {
            if (i < rootPos) {
                leftBack.push_back(back[i]);
                leftMid.push_back(mid[i]);
            }
            else if (i > rootPos)
            {
                rightBack.push_back(back[i-1]);
                rightMid.push_back(mid[i]);
            }
        }
        cur->left = reConstructBinaryTree(leftBack, leftMid);
        cur->right = reConstructBinaryTree(rightBack, rightMid);
        return cur;
    }
    
    //leetcode 309
    int maxProfit(vector<int>& prices) {
        // 最值问题 递推过来的 就是dp
        // 定义dp[i][j] 为第i天 持有状态j的情况下的最大收益
        vector<vector<int>> dp(prices.size(), vector<int>(2));
        // base case j持有状态分别是 持有 还是 没持有
        // 操作 买入 卖出  冷冻期 来让（驱动）这个 状态 转变
        dp[0][0] = 0; //第0天没持有 收益是 0
        dp[0][1] = -prices[0]; //第0天持有 收益 是 -price[0]
        // 状态转移方程
        // dp[i][j] 怎么过来的？ 也就是第i天手里股票持有状态为j的时候的收益是多少？
        // 明显分情况  dp[i][0] dp[i][1]
        /*
        今天不持有股票 分为
            前一天持有股票今天卖出    dp[i][0] = dp[i-1][1] + prices[i]
            前一天没持有股票         dp[i][0] = dp[i-1][0]
        今天持有股票 dp[i][1] = max(dp[i-1][1], -price[i])
            可能是 昨天没有 但我今天买了股票 那么我口袋了少了 dp[i-1][0]-price[i]
            以及前一天持有股票那么 收益平移过来 dp[i-1][1]
        注意 持有股票 然后我卖 相当于这天股价多少我就能有多少钱到我口袋 对不对？这就是这次操作的收益
        */
        /*
        这道题比 股票简单题 多了一个限制 也就是 冷冻期 冷冻期不能买
        那如果我 i-2天没持有 肯定是可以递推过来的
        */
        for (int i = 1; i < prices.size(); i++)
        {
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i]);
            int temp = (i - 2) >= 0 ? dp[i-2][0] : 0;
            dp[i][1] = max(dp[i-1][1], temp - prices[i]);
        }
        return max(dp[prices.size()-1][0], dp[prices.size()-1][1]);
    }
    
    struct SegmentTreeNode
    {
        int start;
        int end;
        int count;

        SegmentTreeNode* left;
        SegmentTreeNode* right;

        SegmentTreeNode(int _start, int _end):start(_start),end(_end) {
            count = 0;
            left = NULL;
            right = NULL;
        }
    };

    SegmentTreeNode* build(int start, int end){
        if (start > end)
            return NULL;
        
        SegmentTreeNode* root = new SegmentTreeNode(start, end);

        if (start == end){
            root->count = 0;
        }else{
            int mid = start + (end - start)/2;
            root->left = build(start, mid);
            root->right = build(mid+1, end);
        }
        return root;
    }

    int count(SegmentTreeNode* root, int start, int end){
        if (root == NULL || start>end)
            return 0;
        if (start == root->start && end == root->end){
            return root->count;
        }
        int mid = root->start + (root->end - root->start)/2;
        int leftcount = 0, rightcount = 0;

        if (start <= mid){
            if (mid < end)
                leftcount = count(root->left, start, mid);
            else
                leftcount = count(root->left, start, end);
        }

        if (mid < end){
            if (start <= mid)
                rightcount = count(root->right, mid+1, end);
            else
                rightcount = count(root->right, start, end);
        }

        return (leftcount + rightcount);
    }

    void insert(SegmentTreeNode* root, int index, int val){
        if (root->start==index && root->end==index){
            root->count += val;
            return;
        }

        int mid = root->start + (root->end - root->start)/2;
        if (index>=root->start && index<=mid){
            insert(root->left, index, val);
        }
        if (index>mid && index<=root->end){
            insert(root->right, index, val);
        }

        root->count = root->left->count + root->right->count;
    }

    vector<int> countSmaller(vector<int>& nums) {
        vector<int> res;
        if (nums.empty())
            return res;
        res.resize(nums.size());
        int start = nums[0];
        int end = nums[0];

        for (int i=1; i<nums.size(); i++){
            start = min(start, nums[i]);
            end = max(end, nums[i]);
        }

        SegmentTreeNode* root = build(start, end);

        for (int i=nums.size()-1; i>=0; i--){
            res[i] = count(root, start, nums[i]-1);
            insert(root, nums[i], 1);
        }

        return res;
    }
    
    // 折纸问题 一张纸 对折一下 展开 凸的痕迹向下 down 如果对折再对折 那么有 三道折痕 分别是 down down up  求对折n次 从上到下 的折痕 依次打印
    // 这里为了简单 用1 表示 down 2 表示 up
    TreeNode* foldPaper(int n)
    {
        // 转换为二叉树问题 对折一下 折痕是根节点 再对折 从上到下的俩新折痕分别是 根节点的 左右节点 依次
        // 技巧是用层序遍历 你想下 每次对折都是上一层的基础上
        TreeNode* root = nullptr;
        for (int i = 0; i < n; i++)
        {
            //1 当前第一次对折
            if (i == 0)
            {
                root = new TreeNode(1);
                continue;
            }
            //2 当前不是第一次对折
            queue<TreeNode*> q;
            q.push(root);
            while (!q.empty()) {
                TreeNode* cur = q.front();
                q.pop();
                if (root->left) {
                    q.push(root->left);
                }
                if (root->right) {
                   q.push(root->right);
                }
                if (root->left == nullptr && root->right == nullptr) {
                    root->left = new TreeNode(1);
                    root->right = new TreeNode(2);
                }
            }
        }
        return root;
    }
    
    //leetcode 120
    int minimumTotal(vector<vector<int>>& triangle) {
        //1. 定义dp[i][j] 为到第i行j索引 的最小路径和
        vector<vector<int>> dp = vector<vector<int>>(triangle.size(), vector<int>(triangle.size()));
        //2. base case
        dp[0][0] = triangle[0][0];
        //3. 状态转移
        //每次向下选择 只能选择 相同 索引 或者 相同索引 + 1
        //所以dp[i][j] = min(dp[i-1][j], dp[i-1][j-1])
        for (int i = 1; i < triangle.size(); i++)
        {
         dp[i][0] = dp[i-1][0] + triangle[i][0]; // 在最左边的时候 只能从上一行最左边移动下来
         for (int j = 1; j < i; j++)
         {
             dp[i][j] = min(dp[i-1][j], dp[i-1][j-1]) + triangle[i][j];
         }
         dp[i][i] = dp[i-1][i-1] + triangle[i][i]; // 在最右边的时候 只能从上一行最右边移动下来
        }
        return *min_element(dp[triangle.size()-1].begin(), dp[triangle.size()-1].end()); //因为最后一行不止一个
     }
    
    //leetcode 1109
    //第 i 条记录 bookings[i] = [i, j, k] 表示在 i 站上车 k 人，乘坐到 j 站，那么在j+1站时候车里需要减去这些人，需要按照车站顺序返回每一站车上的人数
    /*
     通俗解释 记录每个位置上车多少人、下车多少人，上车人数和下车人数的差就是当前车站人数变化，这些人到站后下一站则应减少这些人（注意：坐到终点的人，不需要被减少）
     把每站之前的上车人数累加则可以得出当前车上人数
     */
    vector<int> corpFlightBookings(vector<vector<int>>& bookings, int n) {
        vector<int> res(n + 1, 0);
        for(auto rec : bookings) {//记录每一站点上下车人数 为了返回索引从0开始这里面都索引-1
            res[rec[0] - 1] += rec[2];
            res[rec[1]] -= rec[2];
        }
        for(int i = 1; i < n; i++) {//计算每一站内车上的人数
            res[i] += res[i - 1];
        }
        return vector<int>(res.begin(), res.end() - 1);
    }
    
    //leetcode 1002
    vector<string> commonChars(vector<string>& A) {
        vector<int> nums(26,INT_MAX);
        vector<int> nums_cur;
        int size = A.size();
        for(int i=0;i<size;i++){
            nums_cur = vector<int> (26,0); //弄一个帮助vector记录当前次遍历的数显次数
            for(int j=0;j<A[i].size();j++){
                nums_cur[A[i][j]-'a']++; //记录每个字母出现的次数
            }
            for(int k=0;k<26;k++){
                nums[k] = min(nums[k],nums_cur[k]); // 对比 总的nums里记录的字母出现次数 和 当前遍历num_cur 记录的次数 如果都不为0 肯定取小的如果有为0的那么肯定表示其中一个没这个字符
            }
        }
        vector<string> res;
        string str="";
        for(int k=0;k<26;k++){
            str = "";
            str.push_back((char)(k+'a'));
            while(nums[k]--){
                res.push_back(str);
            }
        }
        return res;
    }
    
    //leetcode 1110
    //删点成林 因为删掉后 独立的子树 就是林 考虑如果删的是带节点的 那么 我们需要对这个节点的父做处理 所以需要一个pre当做parent
    //后续遍历 左右根 最后处理根 符合我们要的操作
    vector<TreeNode*> res;
    set<int> s;
    void postorder(TreeNode *node, TreeNode *pre) {
        if (node == NULL) {
            return;
        }
        postorder(node->left, node);
        postorder(node->right, node);
        if (s.count(node->val)) {
            if (node->left)
                res.push_back(node->left); //要删的是这个节点的父 那么这个节点 这个子树就变成了独立的树 可以推入答案集
            if (node->right)
                res.push_back(node->right);
            if (pre != NULL) { //处理下 parent 节点
                if (pre->left != NULL && pre->left->val == node->val) {
                    pre->left = NULL;
                }
                else if (pre->right != NULL && pre->right->val == node->val) {
                    pre->right = NULL;
                }
            }
        }
    }
    vector<TreeNode*> delNodes(TreeNode* root, vector<int>& to_delete) {
        for (int i = 0; i < to_delete.size(); i++) {
            s.insert(to_delete[i]);
        }
        if (root != NULL && s.count(root->val) == 0) {
            res.push_back(root);
        }
        postorder(root, NULL);
        return res;
    }
    
    //畅通工程案例 一个省20个城市 已经修建好了7条道路 分别连通城市
    /*
     0 1
     6 9
     3 8
     5 11
     2 12
     6 10
     4 8
     问还需要修多少条道路才可以让全部城市都可以连通
     这其实就是并查集 我们最终如果让并查集都在一个分组中 那么所有的城市都在一个树种 就可以了
     */
    int cityRouteBuildNeed()
    {
        UF_Tree_Weighted wftw(20);
        wftw.tounion(0, 1);
        wftw.tounion(6, 9);
        wftw.tounion(3, 8);
        wftw.tounion(5, 11);
        wftw.tounion(2, 12);
        wftw.tounion(6, 10);
        wftw.tounion(4, 8);
        return wftw.groupCnt() - 1;
    }
    
    //leetcode 785
    bool isBipartite(vector<vector<int>>& graph) {
       //graph是邻接矩阵 所以 行数就是节点数  每一行里的元素就是 当前节点的相邻节点
       vector<int> color(graph.size());
       queue<int> q;
       for (int i = 0; i < graph.size(); i++)
       {
           if (color[i] != 0) { //这个可太重要了 染色过的就别看了
               continue;
           }
           q.push(i);
           color[i] = 1;
           while(!q.empty())
           {
               int v  = q.front();
               q.pop();
               vector<int> neighbors = graph[v];
               for (auto n : neighbors)
               {
                   if (color[n] == 0)
                   {
                       q.push(n);
                       color[n] = color[v] == 1 ? 2 : 1;
                   }
                   else
                   {
                       if (color[n] == color[v])
                           return false;
                   }
               }
           }
       }
       return true;
    }
    
    // leetcode 64
    int minPathSum(vector<vector<int>>& grid) {
        //定义dp[i][j]为 走到i行j列的时候的数字最小和
        int row = grid.size();
        int colum = grid[0].size();
        vector<vector<int>> dp(row, vector<int>(colum));

        // base case
        dp[0][0] = grid[0][0];

        // 状态转移方程
        // 每次只能向下 或者 向右  所以
        // dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j];
        // 另外考虑到 第一行 只能从左边过来 第一列只能从上面过来
        // dp[0][j] = dp[0][j-1] + grid[0][j];
        // dp[i][0] = dp[i-1][0] + grid[i][0];
        int i,j;
        for (int i = 1; i < row; i++) {
            dp[i][0] = dp[i-1][0] + grid[i][0];
        }
        for (int j = 1; j < colum; j++) {
            dp[0][j] = dp[0][j-1] + grid[0][j];
        }
        for (i = 1; i < row; i++)
        {
            for (j = 1; j < colum; j++)
            {
                int m = min(dp[i-1][j], dp[i][j-1]);
                dp[i][j] = m + grid[i][j];
            }
            
        }
        return dp[row-1][colum-1];
    }
    
    //leetcode 410
    /*
     给定一个非负整数数组和一个整数 m，你需要将这个数组分成 m 个非空的连续子数组。设计一个算法使得这 m 个子数组各自和的最大值最小。
     这种分割，然后求最的问题的 基本就是 dp  （因为dp就是求最问题的常用解）
     这种分割问题，核心思想是 分割点的枚举 因为分割点事不确定的我们需要 从0-长度这么来枚举
     */
     int splitArray(vector<int>& nums, int m) {
         //1. 要求什么 ？ 子数组各自和的最大值最小 那么我们dp这么定义就行了
         //定义dp为 长度为 i 切割成 j 段 的 子数组中 和 的最小值
         int n = nums.size();
         vector<vector<long long>> dp(n + 1, vector<long long>(m + 1, INT_MAX));
         
         // 补充 因为我们求的是子数组中元素和 所以需要 求个前缀和先
         vector<long long>sum (n + 1, 0);
         for (int i = 1; i <= n; i++) {
             sum[i] = sum[i - 1] + nums[i-1];
         }
         
         //2. base case
         // 很简单的想到 如果只有0个数的时候 那么。。结果没啥意义 是 0
         for (int i = 0; i <= m; i++) {
             dp[0][i] = 0;
         }

         //3.状态转移方程
         // 这种切割问题的核心是 遍历切割点 也就是 dp[i][j] = min(dp[i][j], max(dp[k-1][j-1], sum[i]-sum[k-1])); 因为求的是最小值所以 用min
         //k表示前k个数分为 j-1段 k+1到第i个数分为第j段  来枚举这个 切割点 也就是 切割点k
         for (int i = 1; i <= n; i++) { //遍历n个数
             for (int j = 1; j <= min(m, i); j++) { //遍历分成j段
                 for (int k = j; k <= m; k++) { //j段里 k-1个数 和 k+1到i个数 也就是遍历了所有的情况 i个数j段 然后里面数的按切割点划分 组成的子数组和
                     dp[i][j] = min(dp[i][j], max(dp[k - 1][j - 1], sum[i] - sum[k - 1]));
                 }
             }
         }
         return dp[n][m];
     }
    
    //leetcode64
    /*
     给定一个包含非负整数的 m x n 网格，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
     说明：每次只能向下或者向右移动一步。
     最值问题 状态转移明确 向下 或者 向右
     */
    int minPathSum2(vector<vector<int>>& grid) {
        //1. 定义dp为 走到 i j时候数字总和最小
        int row = grid.size();
        int colum = grid[0].size();
        vector<vector<int>> dp(row + 1, vector<int>(colum + 1));
        
        //2. base case
        dp[0][0] = grid[0][0];
        
        //3. state transform
        for (int i = 1; i < row; i++) {
            dp[i][0] = dp[i-1][0] + grid[i][0];
        }
        
        for (int i = 1; i < colum; i++) {
            dp[0][i] = dp[0][i-1] + grid[0][i];
        }
        
        for (int i = 1; i < row; i++) {
            for (int j = 1; j <colum; j++) {
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j];
            }
        }
        return dp[row-1][colum-1];
    }
    
    //leetcode 剑指offer11
    /*
     把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。例如，数组 [3,4,5,1,2] 为 [1,2,3,4,5] 的一个旋转，该数组的最小值为1。
     */
    int minArray(vector<int>& numbers) {
        int size = numbers.size();
        int low = 0;
        int high = size - 1;
        while (low < high) { //通过二分法 去找 最小值 如果 中间的值 小于 我们末尾的值 表明 最小值 在中间到末尾这段
            int mid = (low + high) / 2;
            if (numbers[mid] < numbers[high]) {
                high = mid;
            }
            else if (numbers[mid] > numbers[high]) // 如果中间的值 大于末尾的值，证明 有序数组前x个数搬到了后面，最小值在后面这段
            {
                low = mid + 1;
            }
            else // 如果中间值和末位置相等，可能有相同数的情况 因为是有序数组所以 --
            {
                high--;
            }
        }
        return numbers[low];
    }
    
    //leetcode 35
    /*
     给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
     你可以假设数组中无重复元素
     */
    int searchInsert(vector<int>& nums, int target) {
        int size = nums.size();
        int low = 0;
        int high = size - 1;

        if (nums[low] > target)
            return 0;
        if (nums[high] < target)
            return high + 1;

        while (low < high) {
            int mid = (low + high) / 2;
            if (nums[mid] < target) {
                low = mid + 1;
            }
            else if (nums[mid] > target)
            {
                high = mid - 1;
            }
            else
            {
                return mid;
            }
        }
        if (nums[low] >= target)
            return low;
        else
            return low + 1;
    }
    
    //leetcode 96 给定一个整数 n，求以 1 ... n 为节点组成的二叉搜索树有多少种？
    /**
        可以先简单的分析下这题，n个连续的数 多少种二叉搜索树？ 二叉搜索树的特点是 左子树都小于根节点 右子树的节点值都大于根节点
        我们简单的枚举 如果 1作为根节点 ，那么 后面n-1个数 有多少种？ 递推到最后 只有 1 个数的时候 这样递推的 可以用动态规划来解决
        这有一个数学公式  卡特兰数
     */
    int numTrees(int n) {
        //1. 定义dp为 i个数时候 二叉搜索树的种数
        vector<int> dp (n + 1, 0);
        //2. base case
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <=n ; i++) {
            for (int j = 1; j <= i; j++) { //可以这么理解 比如 i= 2表示 2个数 然后 j呢是枚举 第一个 到 最后一个数 最为根节点的情况
                dp[i] += dp[j-1] * dp[i - j];
            }
        }
        return dp[n];
    }
};

class NumArray {
public:
    vector<int> sums;

    NumArray(vector<int>& nums) {
        // dp a sum
        sums.resize(nums.size() + 1);
        for (int i = 1; i <= sums.size(); i++)
        {
            sums[i] = sums[i-1] + nums[i-1];
        }
    }
    
    int sumRange(int i, int j) {
        return sums[j] - sums[i];
    }
};

int main(int argc, const char * argv[]) {
    // test deque with stack imp
//    DequeueWithStack dq;
//    dq.push_back(1);
//    dq.push_back(2);
//    dq.push_back(3);
//    std::cout << dq.pop_front() << std::endl;
//    std::cout << dq.pop_front() << std::endl;
//    dq.push_back(4);
//    dq.push_back(5);
//    std::cout << dq.pop_front() << std::endl;
    
    // test stack with queue
//    StackWithQueue sq;
//    sq.push(1);
//    sq.push(2);
//    sq.push(3);
//    cout << sq.pop() << endl;
//    cout << sq.pop() << endl;
//    sq.push(4);
//    sq.push(5);
//    cout << sq.pop() << endl;
//    cout << sq.pop() << endl;
//    cout << sq.pop() << endl;
//    cout << sq.pop() << endl;
//    cout << sq.pop() << endl;

    // test stack with fixed array
//    StackWithFixArray* sa = new StackWithFixArray(5);
//    sa->push(1);
//    sa->push(2);
//    sa->push(3);
//    sa->push(4);
//    sa->push(5);
//    sa->push(6);
//    cout << sa->pop() << endl;
//    cout << sa->pop() << endl;
//    cout << sa->pop() << endl;
//    cout << sa->pop() << endl;
//    cout << sa->pop() << endl;
//    cout << sa->pop() << endl;
    
    // test queue with fixed array
//    QueueWithFixArray qa;
//    qa.push(1);
//    qa.push(2);
//    qa.push(3);
//    qa.push(4);
//    cout << qa.pop() << endl;
//    cout << qa.pop() << endl;
//    cout << qa.pop() << endl;
//    cout << qa.pop() << endl;
//    cout << qa.pop() << endl;
//    qa.push(5);
//    qa.push(6);
//    qa.push(7);
//    qa.push(8);
//    cout << qa.pop() << endl;
//    cout << qa.pop() << endl;
    
    // test min stack
//    MinStack ms;
//    ms.push(1);
//    ms.push(2);
//    ms.push(3);
//    ms.push(4);
//    cout << ms.getMin() << endl;
//    cout << ms.pop() << endl;
//    cout << ms.pop() << endl;
//    cout << ms.pop() << endl;
//    ms.push(0);
//    cout << ms.getMin() << endl;
    
    // test max heap
//    MaxHeap mp;
//    int arr[] = {5, 6, 9, 1, 3, 0, 7};
//    int size = 7;
//    mp.buildMaxHeap(arr, size);
////    mp.CreateHeap(arr, size);
////    mp.heapSort(arr, size);
//    for (int i = 0; i < 7; i++) {
//        cout << arr[i] << endl;
//    }
    
//    Solution sol;
//    sol.isPalindrome("A man, a plan, a canal: Panama");
    
    //test reverse stack with on ly recusive
//    ReverStackWithRecusive rs;
//    stack<int> st;
//    st.push(1);
//    st.push(2);
//    st.push(3);
//    rs.reverse(st);
//    cout << st.top() << endl; st.pop();
//    cout << st.top() << endl; st.pop();
//    cout << st.top() << endl; st.pop();
    
    // test deque window max
//    vector<int> v {4,3,5,4,3,3,6,7};
//    WindowMax wm;
//    vector<int> res = wm.getMaxNumInWindow(v, 3);
    
    // test clockwise print matrix
    Solution sol;
//    vector<int> v1 = {1,2,3,4};
//    vector<int> v2 = {5,6,7,8};
//    vector<int> v3 = {9,10,11,12};
//    vector<vector<int>> v = {v1, v2, v3};
//    sol.spiralOrder(v);
//    vector<int> v{9,8,7,6,5,4,3,2,1,0};
//    sol.plusOne(v);
//    vector<int> v {2,3,1,2,4,3};
//    sol.minSubArrayLen(7, v);
//    vector<int> v = {3,1,2,4};
//    sol.findKthLargest(v, 2);
//    vector<int> v = {1,2,3,2,1};
//    vector<int> v2 = {3,2,1,4,7};
//    sol.findLength(v, v2);
//    vector<int> v {1,1,1,2,2,3};
//    sol.topKFrequent(v, 2);
//    vector<int> v {1,5,9};
//    vector<int> v2 {10,11,13};
//    vector<int> v3 {12,13,15};
//    vector<vector<int>> v4 = {v, v2, v3};
//    sol.kthSmallest(v4, 8);
//    vector<int> v {10,5,4,10,3,1,7,8};
//    sol.lastStoneWeight(v);
//    vector<vector<int>> v(1, vector<int>(1,1));
//    sol.uniquePathsWithObstacles(v);
//    vector<int> v{-2,0,3,-5,2,-1};
//    NumArray na(v);
//    na.sumRange(0, 2);
//    sol.divingBoard(1, 2, 3);
    
//    BSTree<int> bstree;
//    bstree.insert(5);
//    bstree.insert(3);
//    bstree.insert(1);
//    bstree.insert(0);
//    bstree.insert(2);
//    bstree.insert(7);
//    bstree.insert(6);
//    bstree.insert(8);
//    bstree.insert(9);
//    bstree.insert(4);
//
//    bstree.preorder();
//    std::cout << "---" << std::endl;
////    bstree.del(9);
////    bstree.preorder();
////    bstree.del(8);
////    bstree.preorder();
////    bstree.del(7);
////    bstree.preorder();
//    bstree.levelorder();
    
//    vector<int> v {1,2,3,0,2};
//    int val = sol.maxProfit(v);
    
//    vector<int> v {5,2,6,1};
//    sol.countSmaller(v);
    
//    UF uf(5);
//    cout << uf.groupCnt() << endl;
//    cout << uf.connected(1, 2) << endl;
//    uf.tounion(1, 2);
//    cout << uf.groupCnt() << endl;
//    cout << uf.connected(1, 2) << endl;
    
//    UFTree uft(5);
//    cout << uft.groupCnt() << endl;
//    cout << uft.connected(1, 2) << endl;
//    uft.tounion(1, 2);
//    cout << uft.groupCnt() << endl;
//    cout << uft.connected(1, 2) << endl;
    
//    UFTree uftw(5);
//    cout << uftw.groupCnt() << endl;
//    cout << uftw.connected(1, 2) << endl;
//    uftw.tounion(1, 2);
//    cout << uftw.groupCnt() << endl;
//    cout << uftw.connected(1, 2) << endl;
    
//    vector<int> v1 {2};
//    vector<int> v2 {3,4,};
//    vector<int> v3 {6,5,7};
//    vector<int> v4 {4,1,8,3};
//    vector<vector<int>> v {v1, v2, v3, v4};
//    sol.minimumTotal(v);
    
//    SegmentTree segtree;
//    SegmentTreeNode* root = segtree.build(1, 10); //这就相当于把1-10分成1-5 6-10两部分然后 这两部分再递归分割 直到 单独一个
//    segtree.insert(root, 3, 1); //在3 插入1个之后  那么1-10个数就是1 1-5也是1 而6-10个数还是0 ...继续递归到 只有在3索引处 是1
//    cout << segtree.count(root, 1, 4) << endl;
    
//    Trie trie;
//    trie.insert("hello"); //这样就会在根下插入第一个节点h然后接着在h下插入e....
    
    
//    vector<vector<int>> v;
//    v.push_back({1,2,10});
//    v.push_back({2,3,20});
//    v.push_back({2,5,25});
//    sol.corpFlightBookings(v, 5);
    
//    vector<string> v {"bella", "label", "roller"};
//    sol.commonChars(v);

//    Graph g(3);
//    g.addEdge(1, 2);
//    queue<int>* q = g.getAdj(1);
//    queue<int>* q1 = g.getAdj(2);
    
//    Graph g(13);
//    g.addEdge(0, 5);
//    g.addEdge(0, 1);
//    g.addEdge(0, 2);
//    g.addEdge(0, 6);
//    g.addEdge(5, 3);
//    g.addEdge(5, 4);
//    g.addEdge(3, 4);
//    g.addEdge(4, 6);
//
//    g.addEdge(7, 8);
//
//    g.addEdge(9, 11);
//    g.addEdge(9, 10);
//    g.addEdge(9, 12);
//    g.addEdge(11, 12);
//
//    DepthFirstSearch dfs(&g, 0);
//    cout << dfs.getCnt() << endl;
//    cout << dfs.ismakred(5) << endl; //判断5和顶点0是否想通 其实dfs构造里直接就调用了dfs那如果相通肯定就marked了
//    cout << dfs.ismakred(7) << endl;
//
//    BreadFirstSearch bfs(&g, 0);
//    cout << bfs.getCnt() << endl;
//    cout << bfs.ismarked(5) << endl; //判断5和顶点0是否想通 其实bfs构造里直接就调用了dfs那如果相通肯定就marked了
//    cout << bfs.ismarked(7) << endl;
//

//    vector<int> v1 {1,3};
//    vector<int> v2 {0,2};
//    vector<int> v3 {1,3};
//    vector<int> v4 {0,2};
//    vector<vector<int>> v {v1,v2,v3,v4};
//    sol.isBipartite(v);
    
    vector<int> v1 {1,3,1};
    vector<int> v2 {1,5,1};
    vector<int> v3 {4,2,1};
    vector<vector<int>> v {v1, v2, v3};
    sol.minPathSum(v);
    return 0;
};
