**9. Bit Manipulation**

- Bit manipulation problems often require constructing the answer bit by bit/adding the contribution of each bit to the answer. Consider this problem where we build the answer by adding the contribution of each bit: You are given an array $a$ of $n$ non-negative integers $(0 \leq a_{i} \leq 10^{9})$. Determine the value of $\Sigma_{l=1}^{n}\Sigma_{r=l}^{n}f(l,r).(r-l+1)$, where $f(l,r) = a_{l} \oplus a_{l+1} \oplus ... \oplus a_{r}$
```cpp
/*
We assume 1-based indexing of array a. Let's convert a to prefix xor array. We shall evaluate
required answer by adding contribution of each bit. Suppose we are evaluating contribution of
ith bit. We will add the lengths of all subarrays whose xor's ith bit is set. Then the contribution
of ith bit = (1<<i) x (Sum of lengths of subarrays whose xor's ith bit is set) 
*/
for(int i=1;i<=n;i++){
    a[i]^=a[i-1];
}
int ans=0;
for(int i=0;i<32;i++){
    vector<int>sum(n+1,0);
    int ones=0;
    for(int j=0;j<=n;j++){
        if(a[j]&(1<<i)){
            sum[j]=(j?sum[j-1]+j:0);
            ones++;
        }
        else{
            sum[j]=(j?sum[j-1]:0);
        }
    }
    int cnt=0,total=0;
    for(int j=0;j<=n;j++){
        if(a[j]&(1<<i)){
            cnt++;
        }
        else{
            total+=(cnt*j-sum[j])+(sum[n]-sum[j]-(ones-cnt)*j);
        }
    }
    ans+=(1<<i)*total;
}
cout<<ans<<'\n';
```
- To turn off the last bit, do
```cpp
n&=(n-1);
```
- Following code finds $XOR$ of first $n$ natural numbers
```cpp
int x(int n){
    if(n%4==1){
        return 1;
    }
    else if(n%4==2){
        return n+1;
    }
    else if(n%4==3){
        return 0;
    }
    return n;
}
```
- For any two non-negative integers $a$ and $b$, $a + b = (a \oplus b) + 2(a$ & $b)$
- The above equation is derived from two equations: For any two non-negative integers $a$ and $b$,
    - $a + b = (a$ | $b) + (a$ & $b)$
    - $a \oplus b = (a$ | $b) - (a$ & $b)$
- Some useful inequalities
    - $a$ & $b \leq min(a, b)$
    - $a$ | $b \geq max(a, b)$
- Iterating through subsets
```cpp
// Iterating through subsets of {1, 2, 3, ... n}
for(int i=0;i<(1<<n);i++){
    // process subset i
}

// Iterating through subsets with exactly k elements
for(int i=0;i<(1<<n);i++){
    if(__builtin_popcount(i)==k){
        // process subset i
    }
}

// Iterating through subsets of a set x
int i=0;
do{
    // process subset i
}while(i=(i-x)&x);

// Another method of iterating through subsets of a set x
int i=x;
while(i>0){
    // process subset i
    i=(i-1)&x;
}
```
- Iterating through all masks with their submasks
```cpp
for(int m=0;m<(1<<n);m++){
    for(int s=m;s>0;s=(s-1)&m){
        // process submask s of mask m
    }
}
```
- The time complexity of above is $O(3^n)$. The proof is as follows: We will show that the inner loop will execute a total of $O(3^n)$ iterations. If mask $m$ has $k$ enabled bits, then it will have $2^k$ submasks. As we have a total of $\binom{n}{k}$ masks with $k$ enabled bits, therefore total number of submasks (across all masks) would be $\Sigma_{k=0}^{n}\binom{n}{k}2^k = (1+2)^n = 3^n$. Thus the inner loop executes $O(3^n)$ iterations
- We can find minimum xor of two integers in an array by sorting the array and then finding xor between consecutive elements in the array and taking minimum among all the values obtained (It can be shown that two integers in an array having maximum common prefix bits always occur as adjacent elements in the sorted array). Another method to do the same is by using trie

**10. Probability and Expectation**

- A nice problem and its solution\
**Problem**: There are $n$ clients and $n$ servers. Each client sends a request to one of the $n$ servers (with equal probability). If a server gets more than $1$ request, it satisfies only $1$ request. Find expected number of clients whose requests are satisfied.\
\
**Solution**: Let $X_i$ denote the random variable for number of requests satisfied by server $i$. Then by linearity of expectation, answer is $E(X_1 + X_2 + ... + X_n) = E(X_1) + E(X_2) + ... + E(X_n)$. Now, each $E(X_i) = 1 \times (1-(1-\frac{1}{n})^n) + 0 \times (1-\frac{1}{n})^n = 1-(1-\frac{1}{n})^n$. So required answer $= n(1-(1-\frac{1}{n})^n)$\
\
Another similar problem is to find expected number of empty boxes when $n$ balls are randomly placed in $n$ boxes. Here, by linearity of expectation, answer is $n(1-\frac{1}{n})^n$
- Another problem on expectations\
**Problem**: A small pond has a single amoeba living inside it. This particular amoeba has a unique ability, where every minute it can either die, stay alive, split into two or split into three, each with an equal probability. The offsprings produced by this amoeba have the same behaviour, and will act independently of other amoebas. What is the likelihood that this amoeba population will eventually die out?\
\
**Solution**: Let $p$ be the required probability. Then the following equation holds: $p = \frac{1}{4} + \frac{1}{4}p + \frac{1}{4}p^2 + \frac{1}{4}p^3$. Solving this equation, we get $p = 1$ or $p = \sqrt{2} - 1$. $p = 1$ cannot be true and hence $p = \sqrt{2} - 1$ is the required answer
- Another problem on expectations\
**Problem**: A person rolls a fair $p-faced$ die and records the value he rolls. Afterwards, he continues rolling the die until he obtains a value at least as large as the first roll. Let $N$ be the number of rolls after the first he performs. Find $E[N]$\
\
**Solution**: Let the number on the first roll be $x$. Let's calculate expected number of rolls till the experiment ends. Let $y = \frac{x-1}{p}$. Then expected number of rolls $= (1-y)\times1 + y(1-y)\times2 + y^2(1-y)\times3 + ... = \frac{1}{1-y} = \frac{p}{p+1-x}$. Thus, the required answer $= \frac{1}{p}[\Sigma_{x=1}^{p} \frac{p}{p+1-x}] = \Sigma_{i=1}^{p} \frac{1}{i}$
- Another nice problem\
**Problem**: An $n-digit$ number is going to be created by a two-step process. First, an integer $k$ is uniformly selected (where $0 \leq k \leq n$). Then, we select one $n-digit$ number uniformly at random from the collection of all $n-digit$ numbers where digit $d_1$ appears exactly $k$ times and digit $d_2$ appears rest of the $(n-k)$ times ($0 \leq d_1, d_2 \leq 9$). Call this selected number $X$. Determine $E[X]$\
\
**Solution**: First, the integer $k$ is chosen with a probability of $\frac{1}{n+1}$. Then, the number $X$ is chosen with a probability of $\frac{1}{\frac{n!}{k!(n-k)!}}$ (since there are $\frac{n!}{k!(n-k)!}$ numbers satisfying the given criteria). Thus, by definition of $E[X]$, we have
$$E[X] = \frac{1}{n+1}\sum_{k=0}^{n}\frac{k!(n-k)!}{n!} \times \textrm{Sum of numbers satisfying the given criteria with parameter }k$$
Now, the term involving the sum of numbers can be evaluated by adding contribution of each bit (with base $10$ ofcourse)
$$\textrm{Contribution of }i^{th} bit = 10^{i} \times \big[\frac{d_1.(n-1)!}{(k-1)!(n-k)!} + \frac{d_2.(n-1)!}{k!(n-k-1)!}\big] = 10^{i} \times \big[\frac{(n-1)!}{k!(n-k)!}\big(d_1.k + d_2.(n-k)\big)\big]$$
Thus, we have
$$\textrm{Sum of numbers satisfying the given criteria with parameter }k = \sum_{i=0}^{n-1}10^{i} \times \big[\frac{(n-1)!}{k!(n-k)!}\big(d_1.k + d_2.(n-k)\big)\big] = \frac{10^n-1}{9} \times \big[\frac{(n-1)!}{k!(n-k)!}\big(d_1.k + d_2.(n-k)\big)\big]$$
Thus, we arrive at
$$E[X] = \frac{1}{n+1}\sum_{k=0}^{n}\frac{k!(n-k)!}{n!} \times \frac{10^n-1}{9} \times \big[\frac{(n-1)!}{k!(n-k)!}\big(d_1.k + d_2.(n-k)\big)\big] = \frac{10^n-1}{9n(n+1)}\sum_{k=0}^{n}(d_2n + (d_1 - d_2)k) = \frac{10^n-1}{9n(n+1)} \times (d_2n(n+1) + (d_1 - d_2)\frac{n(n+1)}{2})$$
Thus, finally we get our required answer
$$E[X] = \big(\frac{10^n-1}{9}\big)\big(\frac{d_1 + d_2}{2}\big)$$

**11. Monotonic Stack**

- Next greater element for each element of the array
```cpp
stack<int>stk;
vector<int>nge(n,-1); // nge[i] -> index of next greater element / -1 if no next greater element
for(int i=0;i<n;i++){
    while(!stk.empty() && a[i]>a[stk.top()]){
        nge[stk.top()]=i;
        stk.pop();
    }
    stk.push(i);
}
```
- Can be modified to find next smaller/previous greater/previous smaller element. For finding previous elements, traverse the array in reverse direction. For finding smaller elements, the monotonicity of stack can be reversed
- Suppose we have to find previous greater element to the left of previous smaller element for each element of the array. This can be done as follows
```cpp
stack<int>stk,stk2;
vector<int>pge(n,-1); // pge[i] -> Previous greater element to the left of previous smaller element for index i
for(int i=n-1;i>=0;i--){
    while(!stk.empty() && a[stk.top()]<a[i]){
        pge[stk.top()]=i;
        stk.pop();
    }
    while(!stk2.empty() && a[stk2.top()]>a[i]){
        if(pge[stk2.top()]==-1){
            stk.push(stk2.top());
        }
        stk2.pop();
    }
    stk2.push(i);
}
```
- Monotonic stack can be used to find area of largest rectangle in a histogram
```cpp
// h[i] -> Height of ith bar
stack<int>stk;
int max_area=0;
for(int i=0;i<n;i++){
    if(stk.empty() || h[stk.top()]<=h[i]){
        stk.push(i);
    }
    else{
        int top=stk.top();
        stk.pop();
        int area=h[top]*(stk.empty()?i:i-stk.top()-1); // Area of rectangle with h[top] as smallest bar
        max_area=max(max_area,area);
    }
}
while(!stk.empty()){
    int top=stk.top();
    stk.pop();
    int area=h[top]*(stk.empty()?n:n-stk.top()-1);
    max_area=max(max_area,area);
}
cout<<max_area<<'\n';
```

**12. KMP Algorithm**

- First, we construct $lps$ array for the string $s$
```cpp
void compute_lps(string& s,vector<int>& lps){
    int n=s.length();
    lps.resize(n,0);
    // lps[i] -> Longest proper prefix of s[0...i] which is also a suffix of s[0...i]. Thus lps[i] <= i
    int len=0;
    lps[0]=0; // Obvious
    int i=1;
    while(i<n){
        if(s[i]==s[len]){
            len++;
            lps[i]=len; // Obvious
            i++;
        }
        else{
            if(len){
                len=lps[len-1]; // Explained below
            }
            else{
                lps[i]=0; // Obvious
                i++;
            }
        }
    }
}
```
- Suppose we have processed string $s$ (assuming 0-indexing) till index $i-1$\
  $s_0s_1...s_{len-1}...s_{i-len}s_{i-len+1}...s_{i-1}$ (Here $len = lps[i-1])$\
  Here, we have $s_0s_1...s_{len-1} = s_{i-len}s_{i-len+1}...s_{i-1}$ (Let's call it equation $\oplus$). Now, suppose $s_i \neq s_{len}$\
  We need to change $len$ to maximum possible $len^{'}$ such that $s_0s_1...s_{len^{'}-1} = s_{i-len^{'}}s_{i-len^{'}+1}...s_{i-1}$ (Call it equation $\odot$). Note that $len^{'} > len$ cannot happen since $lps[i-1] = len$. Thus we need to change (decrease) $len$ to maximum possible $len^{'}$ such that $len^{'} < len$ and satisfies equation $\odot$\
  Note that $s_{i-len^{'}}s_{i-len^{'}+1}...s_{i-1} = s_{len-len^{'}}s_{len-len{'}+1}...s_{len-1}$ (By equation $\oplus$ and the fact that $len^{'} < len$). Thus, we have\
  $s_0s_1...s_{len^{'}-1} = s_{len-len^{'}}s_{len-len{'}+1}...s_{len-1}$ (By previous equation and $\odot$)\
  Above equation tells us that $len^{'}$ is the largest value $< len$ such that $s_0s_1...s_{len^{'}-1} = s_{len-len^{'}}s_{len-len{'}+1}...s_{len-1}$\
  Note that $len^{'}$ satisfies the definition of $lps[len-1]$. That is why we change $len$ to $lps[len-1]$
- Now, we can run KMP algorithm to find all indices where string $pat$ appears as a substring in string $txt$
```cpp
vector<int> KMP(string& pat,string& txt){
    int m=pat.length();
    int n=txt.length();
    vector<int>lps(m);
    vector<int>ans;
    compute_lps(pat,lps);
    while(n-i>=m-j){
        if(pat[j]==txt[i]){
            j++;
            i++;
        }
        if(j==m){
            ans.push_back(i-j);
            j=lps[j-1];
        }
        else if(i<n && pat[j]!=txt[i]){
            if(j){
                j=lps[j-1];
            }
            else{
                i++;
            }
        }
    }
    return ans;
}
```
- $lps$ array can be used to find longest palindromic prefix of a string $s$. Suppose $len(s) = n$. We concatenate $s$ with $rev(s)$. Now, we compute $lps$ array for this new string. The value of $lps[2n-1]$ (assuming 0-indexing) would be the answer

**13. Trie**

```cpp
struct node{
    struct node* children[26];
    bool end;
};

struct node* get_node(){
    struct node* n=new node;
    for(int i=0;i<26:i++){
        n->children[i]=NULL;
    }
    n->end=false;
    return n;
}

void insert(struct node* root,string s){
    struct node* curr=root;
    for(int i=0;i<s.length();i++){
        int idx=s[i]-'a';
        if(!curr->children[idx]){
            curr->children[idx]=get_node();
        }
        curr=curr->children[idx];
    }
    curr->end=true;
}

bool search(struct node* root,string s){
    struct node* curr=root;
    for(int i=0;i<s.length();i++){
        int idx=s[i]-'a';
        if(!curr->children[idx]){
            return false;
        }
        curr=curr->children[idx];
    }
    return curr->end;
}
```
- Code can be modified to insert array elements as well
- $search()$ can be modified to return number of matching characters
- Can be used to solve problems related to finding $maximum/minimum$ $xor/xnor$ of two integers in an array

**14. Fenwick Tree (Binary Indexed Tree)**

```cpp
int sum(int k){
    // For finding sum from index 1 to index k (1 based indexing)
    int sum=0;
    while(k>0){
        sum+=tree[k];
        // tree[k] denotes sum of array elements of subarray [k-(k&-k)+1 to k]. The length of the subarray = greatest power of 2 which divides k
        k-=(k&-k);
        // Can also do k&=(k-1)
    }
    return sum;
}

void add(int k,int x){
    // For adding x to element at index k
    while(k<=n){
        tree[k]+=x;
        k+=(k&-k);
    }
}
```

**15. Segment Tree**

- Can support all range queries where it is possible to divide a range into two parts, calculate the answer separately for both parts and then efficiently combine the answers. Example of such queries are $minimum$ and $maximum$, $greatest$ $common$ $divisor$, and bit operations $and$, $or$ and $xor$
- Consider this template for calculating $sums$ in an array using segment tree
```cpp
// Note that lx to rx-1 represents current segment and current index in segment tree array is x
// We are calculating sum from index l to r-1
struct segtree{
    int sz;
    vector<int>sums;
    
    void init(int n){
        sz=1;
        while(sz<n){
            sz<<=1;
        }
        sums.assign(2*sz,0);
    }
    
    void build(vector<int>& a,int x,int lx,int rx){
        if(rx==lx+1){
            if(lx<(int)a.size()){
                sums[x]=a[lx];
            }
            return;
        }
        int m=lx+(rx-lx)/2;
        build(a,2*x+1,lx,m);
        build(a,2*x+2,m,rx);
        sums[x]=sums[2*x+1]+sums[2*x+2];
    }
    
    void set(int i,int v,int x,int lx,int rx){
        if(rx==lx+1){
            sums[x]=v;
            return;
        }
        int m=lx+(rx-lx)/2;
        if(i<m){
            set(i,v,2*x+1,lx,m);
        }
        else{
            set(i,v,2*x+2,m,rx);
        }
        sums[x]=sums[2*x+1]+sums[2*x+2];
    }
    
    int sum(int l,int r,int x,int lx,int rx){
        if(lx>=r || rx<=l){
            return 0;
        }
        if(lx>=l && rx<=r){
            return sums[x];
        }
        int m=lx+(rx-lx)/2;
        int s1=sum(l,r,2*x+1,lx,m);
        int s2=sum(l,r,2*x+2,m,rx);
        return s1+s2;
    }
};
```
- General segment tree template looks something like this
```cpp
// Note that lx to rx-1 represents current segment and current index in segment tree array is x
struct item{

};

struct segtree{
    int sz;
    vector<item>values;

    item NEUTRAL_ELEMENT=;

    item merge(item& a,item& b){
    	// Merged value of item a and item b
    }

    item single(int v){
        
    }
    
    void init(int n){
        sz=1;
        while(sz<n){
            sz<<=1;
        }
        values.resize(2*sz);
    }
    
    void build(vector<int>& a,int x,int lx,int rx){
        if(rx==lx+1){
            if(lx<(int)a.size()){
                values[x]=single(a[lx]);
            }
            return;
        }
        int m=lx+(rx-lx)/2;
        build(a,2*x+1,lx,m);
        build(a,2*x+2,m,rx);
        values[x]=merge(values[2*x+1],values[2*x+2]);
    }

    void build(vector<int>& a){
        build(a,0,0,sz);
    }
    
    void set(int i,int v,int x,int lx,int rx){
        if(rx==lx+1){
            values[x]=single(v);
            return;
        }
        int m=lx+(rx-lx)/2;
        if(i<m){
            set(i,v,2*x+1,lx,m);
        }
        else{
            set(i,v,2*x+2,m,rx);
        }
        values[x]=merge(values[2*x+1],values[2*x+2]);
    }

    void set(int i,int v){
        set(i,v,0,0,sz);
    }
    
    item calc(int l,int r,int x,int lx,int rx){
        if(lx>=r || rx<=l){
            return NEUTRAL_ELEMENT;
        }
        if(lx>=l && rx<=r){
            return values[x];
        }
        int m=lx+(rx-lx)/2;
        item x1=calc(l,r,2*x+1,lx,m);
        item x2=calc(l,r,2*x+2,m,rx);
        return merge(x1,x2);
    }

    item calc(int l,int r){
        return calc(l,r,0,0,sz);
    }
};
```
- We can also binary search on the segment tree (By not traversing those nodes for which we are certain about some property and traversing only the relevant nodes). For example, suppose in a binary array, we have to calculate the index of $k^{th}$ $1$ where flipping of elements at a particular index is also supported. This can be done by building a segment tree on sum of segments and finding the first index where sum is $\geq k$
- Can be used to solve problems involving determining count of nested intervals for each interval, determining count of inversions for each element of a permutation, etc
- We can solve the problem of finding count of elements $\leq x$ (or $\geq x)$ in a subarray $[l...r]$ by using segment tree. We build a segment tree and store the maximum and minimum values for each segment. Now, we traverse the tree recursively. We can stop recursion at three types of nodes:
    - Nodes for which minimum value is $> x$ (We simply return when this node is encountered)
    - Nodes for which maximum value is $\leq x$ (Here, we return from this node and add the length of the segment represented by this node to the answer)
    - Nodes which do not intersect with the segment $[l...r]$ (We simply return when this node is encountered)
- **Mass changes for associative and commutative operations**: Suppose we have an array $a$ of $n$ elements and we want to process following operations:\
&nbsp;&nbsp; $(1)$ $modify(l, r, v): a_i = a_i \otimes v$ for all $l \leq i < r$ ($\otimes$ is an associative and commutative operation)\
&nbsp;&nbsp; $(2)$ $get(i):$ get the value of $a_i$\
The above operations can be performed by building a segment tree where each node stores the operation to perform on the segment represented by that node
```cpp
struct segtree{
    int sz;
    int NO_OPERATION=INT_MAX;
    vector<int>operations;

    int operation(int a,int b){
        if(b==NO_OPERATION){
            return a;
        }
    }
    
    void apply_operation(int& a,int b){
        a=operation(a,b);
    }

    void init(int n){
        sz=1; 
        while(sz<n) sz<<=1; 
        operations.resize(2*sz);
    }
    
    void modify(int l,int r,int v,int x,int lx,int rx){
        if(lx>=r || rx<=l) return;
        if(lx>=l && rx<=r){
            apply_operation(operations[x],v);
            return;
        }
        int m=lx+(rx-lx)/2;
        modify(l,r,v,2*x+1,lx,m);
        modify(l,r,v,2*x+2,m,rx);
    }
    
    void modify(int l,int r,int v){
        modify(l,r,v,0,0,sz);
    }
    
    int get(int i,int x,int lx,int rx){
        if(rx==lx+1){
            return operations[x];
        }
        int m=lx+(rx-lx)/2;
        int res;
        if(i<m){
            res=get(i,2*x+1,lx,m);
        }
        else{
            res=get(i,2*x+2,m,rx);
        }
        return operation(res,operations[x]);
    }
    
    int get(int i){
        return get(i,0,0,sz);
    }
};
```
- **Mass changes for associative and non-commutative operations**: Suppose instead the operation $\otimes$ is non-commutative. Then we need to ensure that the older operation is further down in the segment tree. This can be ensured by lazy propagation
```cpp
struct segtree{
    int sz;
    int NO_OPERATION=INT_MAX;
    vector<int>operations;

    int operation(int a,int b){
        if(b==NO_OPERATION){
            return a;
        }
    }
    
    void apply_operation(int& a,int b){
        a=operation(a,b);
    }

    void init(int n){
        sz=1; 
        while(sz<n) sz<<=1; 
        operations.resize(2*sz);
    }
    
    void propagate(int x,int lx,int rx){
        if(rx==lx+1){
            return;
        }
        apply_operation(operations[2*x+1],operations[x]);
        apply_operation(operations[2*x+2],operations[x]);
        operations[x]=NO_OPERATION;
    }
    
    void modify(int l,int r,int v,int x,int lx,int rx){
        propagate(x,lx,rx);
        if(lx>=r || rx<=l) return;
        if(lx>=l && rx<=r){
            apply_operation(operations[x],v);
            return;
        }
        int m=lx+(rx-lx)/2;
        modify(l,r,v,2*x+1,lx,m);
        modify(l,r,v,2*x+2,m,rx);
    }
    
    void modify(int l,int r,int v){
        modify(l,r,v,0,0,sz);
    }
    
    int get(int i,int x,int lx,int rx){
        propagate(x,lx,rx);
        if(rx==lx+1){
            return operations[x];
        }
        int m=lx+(rx-lx)/2;
        int res;
        if(i<m){
            res=get(i,2*x+1,lx,m);
        }
        else{
            res=get(i,2*x+2,m,rx);
        }
        return res;
    }
    
    int get(int i){
        return get(i,0,0,sz);
    }
};
```
- [Codeforces EDU - Segment Tree, part 1](https://codeforces.com/edu/course/2/lesson/4)
- [Codeforces EDU - Segment Tree, part 2](https://codeforces.com/edu/course/2/lesson/5)
- [CP-Algorithms - Segment Tree](https://cp-algorithms.com/data_structures/segment_tree.html)

**16. Heap**

- Following is implementation for MinHeap. Similarly MaxHeap can be implemented
```cpp
struct minheap{
    int sz;
    vector<int>heap;
    
    void build_heap(vector<int>& a){
        int n=a.size();
        sz=n;
        heap.resize(n);
        for(int i=0;i<n;i++){
            heap[i]=a[i];
        }
        for(int i=n/2-1;i>=0;i--){
            min_heapify(i);
        }
    }
    
    void min_heapify(int i){
        int l=2*i+1,r=2*i+2;
        int smallest=i;
        if(l<sz && heap[l]<heap[i]){
            smallest=l;
        }
        if(r<sz && heap[r]<heap[smallest]){
            smallest=r;
        }
        if(smallest!=i){
            swap(heap[i],heap[smallest]);
            min_heapify(smallest);
        }
    }
    
    void decrease_key(int i,int val){
        heap[i]=val;
        while(i && heap[(i-1)/2]>heap[i]){
            swap(heap[i],heap[(i-1)/2]);
            i=(i-1)/2;
        }
    }
};
```

**17. ordered_set and ordered_multiset**

```cpp
#include<ext/pb_ds/assoc_container.hpp> // Common file
#include<ext/pb_ds/tree_policy.hpp> // Including tree_order_statistics_node_update
using namespace __gnu_pbds;

#define ordered_set tree<int,null_type,less<int>,rb_tree_tag,tree_order_statistics_node_update> // Ordered Set
#define ordered_multiset tree<int,null_type,less_equal<int>,rb_tree_tag,tree_order_statistics_node_update> // Ordered Multiset

/*
    ordered_set st; ordered_multiset mst;
    st.order_of_key(k) -> Finds index of integer k in the set (or say count of integers strictly less than k in the set)
    mst.order_of_key(k) -> Finds index of integer k in the multiset (or say count of integers strictly less than k in the multiset)
    *st.find_by_order(i) -> Finds element at index i in the set
    *mst.find_by_order(i) -> Finds element at index i in the multiset
*/
```

**18. General notes, techniques and ideas**

- Sorting intervals as per starting/ending time for an efficient algorithm
- Median plays an important role in some problems
- Some problems require us to think in a different way. For example, traversing the array in backward direction, simulating the opposite operation mentioned in the problem, etc
- Many problems require some careful observation to solve it. For example, observations related to parity, binary / $n$-ary representation of a number and some invariants/monovariants
- Many problems use the concept used in Huffman Coding (choosing $k$ maximum / $k$ minimum values and combining them). It is implemented using priority queue
- In constructive problems, we often have to prove an upper bound (or a lower bound) on some property and then try to come up with a construction that achieves that bound
- In many problems, a multiset can work as a priority queue. It not only supports finding minimum and maximum elements but also removing a particular element from the multiset. The time complexity of all those operations is $O(log$ $n)$
- A multiset can be used to simulate problems involving a timeline. Each event happening at a particular time can be inserted into the multiset and it can be processed
- In many problems, two instances of same data structure are required to simulate the process mentioned in the problem with better complexity. For example, using two instances of multiset/priority queue to maintain lower and upper half of sorted elements, etc.
- The constraints of the problem provide helpful information. For example, many problems involving Bitmask DP have extremely small constraints to allow programs having exponential time complexity. Problems having very large constraints often involve binary search ($O(log$ $n)$ complexity) or some $O(1)$ computation
- Bitsets can be used to reduce running time of many algorithms by a constant factor (equal to word size which is usually $32$ or $64$)
- Many counting problems, like counting pairs of elements/counting subarrays satisfying some property can be solved using divide and conquer approach (If common techniques like fixing the $L$ pointer or $2$ pointer method doesn't work)
- Quickselect is an algorithm to find $k^{th}$ smallest element in an array. It is based on Lomuto Partition technique of Quicksort algorithm
```cpp
void partition(vector<int>& a,int l,int r){
    int i=l;
    for(int j=l;j<=r-1;j++){
        if(a[j]<=a[r]){
            swap(a[j],a[i++]):
        }
    }
    swap(a[i],a[r]);
    return i;
}

int quickselect(vector<int>& a,int l,int r,int k){
    if(k>0 && k<=r-l+1){
        int idx=partition(a,l,r);
        if(idx-l==k-1){
            return a[idx];
        }
        if(idx-l>k-1){
            return quickselect(a,l,idx-1,k);
        }
        return quickselect(a,idx+1,r,k-idx+l-1);
    }
    return INT_MAX;
}
```
- Below code finds number of inversions in an array in $O(n$ $log$ $n)$ (By variation of merge sort)
```cpp
int num_of_inversions(int start,int end,vector<int>& a){
    if(start>=end){
        return 0;
    }
    int ans=0;
    int mid=start+(end-start)/2;
    ans+=num_of_inversions(start,mid,a);
    ans+=num_of_inversions(mid+1,end,a);
    int ans2=ans;
    vector<int>b(end-start+1);
    int i=start,j=mid+1,k=0;
    while(i<=mid && j<=end){
        if(a[i]<=a[j]){
            b[k++]=a[i++];
            ans2+=(j-mid-1);
        }
        else{
            b[k++]=a[j++];
            ans+=(mid-i+1);
        }
    }
    while(i<=mid){
        b[k++]=a[i++];
        ans2+=(end-mid);
    }
    while(j<=end){
        b[k++]=a[j++];
    }
    for(i=start;i<=end;i++){
        a[i]=b[i-start];
    }
    // Note that ans == ans2
    return ans;
}
```
- Below code finds maximum subarray sum for subarrays whose length lies between $l$ and $r$
```cpp
vector<int>presum(n+1,0);
int ans=INT_MIN;
multiset<int>mst;
for(int i=1;i<=n;i++){
    presum[i]=presum[i-1]+a[i]; // Assuming array elements are stored in a[1...n] and a[0] = 0
    if(i>=l){
        mst.insert(presum[i-l]);
    }
    if(i>r){
        mst.erase(mst.find(presum[i-r-1]));
    }
    if(!mst.empty()){
        ans=max(ans,presum[i]-*(mst.begin()));
    }
}
cout<<ans<<'\n';
// Similar idea can be used to find minimum subarray sum for subarrays whose length lies between l and r
```
- Suppose we are given an array of integers and we have to find minimum number of strictly decreasing sequences in it. This can be accomplished as follows
```cpp
int minimum_decreasing_sequences(vector<int>& a){
    int n=a.size();
    multiset<int>mst;
    for(int i=0;i<n;i++){
        auto itr=mst.upper_bound(a[i]);
        if(itr==mst.end()){
            mst.insert(a[i]);
        }
        else{
            mst.erase(itr);
            mst.insert(a[i]);
        }
    }
    return (int)mst.size();
}
```
- Suppose we have to find maximum number of elements in subarray $a[0...i]$ which have sum $\leq x$ for each $i$ from $0$ to $n - 1$. This can be accomplished as follows
```cpp
vector<int>ans(n);
priority_queue<int>pq;
int sum=0;
for(int i=0;i<n;i++){
    pq.push(a[i]);
    sum+=a[i];
    if(sum>x){
        sum-=pq.top();
        pq.pop();
    }
    ans[i]=(int)pq.size();
}
```
- Suppose we have to find maximum sum subsequence of size $\leq m$ in the range $[0,i]$ for each $i$ from $0$ to $n - 1$. This can be done as follows
```cpp
vector<int>ans(n);
priority_queue<int>pq;
int sum=0;
for(int i=0;i<n;i++){
    if(a[i]>0){
        pq.push(a[i]);
        sum+=a[i];
    }
    if((int)pq.size()>m){
        sum-=pq.top();
        pq.pop();
    }
    ans[i]=sum;
}
```
- Many problems have solutions based on some pattern/observation. These can be proved by Mathematical Induction. As an example, suppose an array $x$ of $n$ integers is given. You need to find out smallest integer $s$ such that $s$ cannot be written as sum of a subset of integers in $x$. Check the following approach to find $s$
```cpp
/*
Let us assume that we have processed till index i in the sorted array and we can write
every integer from 1 to sum[1..i] (sum till index i) as a sum of subset of x[1..i]. Now,
if x[i+1] <= sum[1..i]+1, then we can write every number from sum[1..i]+1 to sum[1..i]+x[i+1]
by using x[i+1] and writing the remaining sum from subset of x[1..i]. Also, we can write
any number from 1 to sum[1..i] by using subset of x[1..i] (By assumption). Thus, we can
write all integers from 1 to sum[1..(i+1)] as sum of elements of subset of x[1..(i+1)].
Thus, by induction, all integers from 1 to sum[1..i] can be written as sum of elements of
subset of x[1..i], if x[i]<=sum[1..(i-1)]+1. If x[i]>sum[1..(i-1)]+1, then we can never write
the number p = sum[1..(i-1)]+1 as sum of elements of subset of x[1..i]. Thus answer would
be p
*/
int smallest_sum(vector<int>& x,int n){
    sort(x.begin(),x.end());
    if(x[0]>1){
        return 1;
    }
    int sum=0;
    for(int i=0;i<n;i++){
        if(x[i]>sum+1){
            return sum+1;
        }
        sum+=x[i];
    }
    return sum+1;
}
```
- Suppose an array of positive integers is given. We need to calculate $k^{th}$ smallest sum that can be made from these integers where each integer can be chosen any number of times. If multiple sets have same sum, the sum is counted only once. This is done as follows
```cpp
set<int>st;
st.insert(0);
for(int i=0;i<k;i++){
    int mn=*st.begin();
    st.erase(mn);
    for(int j=0;j<n;j++){
        st.insert(mn+a[j]);
    }
}
cout<<*st.begin()<<'\n';
// If multiple sets having same sum were counted differently, then use multiset instead of set
```
- Checkout the following method to calculate LIS (Longest Increasing Subsequence) in an array in $O(n$ $log$ $n)$ time
```cpp
vector<int>dp(n);
for(int i=0;i<n;i++){
    auto it=lower_bound(dp.begin(),dp.end(),a[i]);
    if(it==dp.end()){
        dp.push_back(a[i]);
    }
    else{
        *it=a[i];
    }
}
cout<<(int)dp.size()<<'\n';
```
