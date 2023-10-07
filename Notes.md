# Notes

## For theory, refer the following links:
- [Codeforces Catalog](https://codeforces.com/catalog)
- [Codeforces EDU](https://codeforces.com/edu/courses)
- [CP-Algorithms](https://cp-algorithms.com/)
- [Topcoder - Competitive Programming](https://www.topcoder.com/thrive/tracks?track=Competitive%20Programming)
- [USACO Guide](https://usaco.guide/)
  
**1. Binary Search**
- General code looks something like this
```cpp
int lo=LOW,hi=HIGH,mid,ans=-1;
while(lo<=hi){
    mid=lo+(hi-lo)/2;
    if(good(mid)){
        ans=mid;
        lo=mid+1; // or hi=mid-1;
    }
    else{
        hi=mid-1; // or lo=mid+1;
    }
}
```
- Another way of implementing binary search: Suppose we are given the following:
    - $[l,r]$: a range of integers
    - $f$: a function from integers to booleans which satisfies following property: there exists some integer $t$ such that for all $l \leq x < t$, $f(x)$ is true and for all $t \leq x \leq r$, $f(x)$ is false ($f$ is called a predicate)
```cpp
int lo=l-1,hi=r+1,mid;
// [l,lo] consists of true values, [hi,r] consists of false values, [lo+1,hi-1] is unexplored
while(hi-lo>1){
    mid=lo+(hi-lo)/2;
    if(f(mid)){
        lo=mid;
    }
    else{
        hi=mid;
    }
    return hi; // first false
    // return lo for last true
}
```
- Binary search to search elements in sorted arrays
- Binary search on answer
- Often used in minimax/maximin problems
- [Codeforces EDU - Binary Search](https://codeforces.com/edu/course/2/lesson/6)

**2. Two Pointers Method**

- General code looks something like this
```cpp
int l=0;
for(int r=0;r<n;r++){
    add(a[r]);
    while(!good()){
        remove(a[l++]);
    }
}
```
- Used for maintaining good segments
- Utilizes the fact that if a property holds for shorter segment (longer segment), then it must also hold for longer segment (shorter segment)
- Other problems that can be solved using this approach include Trapping Rainwater, Sliding window maximum/minimum, checking whether there exist two elements with a particular sum, etc
- [Codeforces EDU - Two Pointers Method](https://codeforces.com/edu/course/2/lesson/9)

**3. Prefix sums, Prefix xors, 2D Prefix sums and Difference Arrays**

- For $1D$ Prefix Sum
```cpp
pre[0]=a[0];
for(int i=1;i<n;i++){
    pre[i]=pre[i-1]+a[i];
}
// Sum of array elements in range [l,r] = pre[r]-(l?pre[l-1]:0)
```
- For $1D$ Prefix XOR
```cpp
prex[0]=a[0];
for(int i=1;i<n;i++){
    prex[i]=prex[i-1]^a[i];
}
// XOR of array elements in range [l,r] = prex[r]^(l?prex[l-1]:0)
```
- For $2D$ Prefix Sum
```cpp
for(int i=0;i<n;i++){
    for(int j=0;j<m;j++){
        if(!i){
            if(!j){
                pre[i][j]=a[i][j];
            }
            else{
                pre[i][j]=pre[i][j-1]+a[i][j];
            }
        }
        else{
            if(!j){
                pre[i][j]=pre[i-1][j]+a[i][j];
            }
            else{
                pre[i][j]=pre[i-1][j]+pre[i][j-1]-pre[i-1][j-1]+a[i][j];
            }
        }
    }
}
// Sum of array elements from [l1,r1] to [l2,r2] = pre[l2][r2]-(r1?pre[l1][r1-1]:0)-(l1?pre[l1-1][r1]:0)+(l1 && r1?pre[l1-1][r1-1]:0)
```
- Difference Array is used to update a range and retrieve value at a single point. For example consider the queries, $(1)$ Add $x$ to $[a, b]$ $(2)$ Find value at position $p$. For this, we can create a difference array such that first element of difference array = first element of given array and subsequent elements of difference array = difference between consecutive array elements of original one. To process first type of query, just add $x$ at position $a$ and subtract $x$ from position $b + 1$ in difference array. To process second type of query, we need to add elements of difference array from start till position $p$. This can be done using a Fenwick Tree/Segment Tree
- [Codeforces EDU - Prefix sums and difference arrays](https://codeforces.com/edu/course/3/lesson/10)

**4. Disjoint Set Union (DSU)**

- Data structures and Initialization
```cpp
vector<int>p(n),r(n);
for(int i=0;i<n;i++){
    p[i]=i;
    r[i]=1;
}
```
- Find function
```cpp
int get(int x,vector<int>& p){
    return p[x]=(p[x]==x?x:get(p[x],p));
}
```
- Union function
```cpp
void unite(int x,int y,vector<int>& p,vector<int>& r){
    x=get(x,p);
    y=get(y,p);
    if(x==y){
        return;
    }
    if(r[x]==r[y]){
        r[x]++;
    }
    if(r[x]>r[y]){
        p[y]=x;
        // For any associative and commutative function, we can do something like this
        // sum[x]+=sum[y];
        // min[x]=min(min[x],min[y]);
        // max[x]=max(max[x],max[y]);
    }
    else{
        p[x]=y;
        // For any associative and commutative function, we can do something like this
        // sum[y]+=sum[x];
        // min[y]=min(min[x],min[y]);
        // max[y]=max(max[x],max[y]);
    }
}
```
- Sometimes, it's better to avoid path compression
- Sometimes, we need to traverse the queries in reverse order when the operation is of "cut" (reverse of "join")
- Sometimes, it's better to avoid union by rank.
- Can be used to solve "Problem with people" types of problems. There are two types of queries: $(1)$ The person at position $i$ goes away and $(2)$ Find the nearest person to the right for position $p$ that did not leave
- [Codeforces EDU - Disjoint Sets Union](https://codeforces.com/edu/course/2/lesson/7)

**5. Dynamic Programming (DP)**

- In Dynamic Programming, we build the DP array as per the recurrence relation we have. As an example, consider the problem in which you are given $q$ queries of type $+ x$ or $- x$. A query of type $+ x$ adds $x$ to the multiset (which is initially empty) and a query of type $- x$ removes a single instance of $x$ from the multiset (It is guaranteed that $x$ exists in the multiset when a query of type $- x$ appears). You are also given an integer $k$. After each query, you need to output the number of ways to obtain $k$ as sum of some numbers in the multiset
```cpp
vector<int>dp(k+1,0); // dp[i] -> Number of ways to obtain i as sum of some numbers in the multiset
dp[0]=1;
while(q--){
    char op;
    cin>>op;
    int x;
    cin>>x;
    if(op=='+'){
        // We need to update the value of dp[i] as dp[i]+=dp[i-x] where dp[i-x] is its old value (not the updated one)
        for(int i=k;i>=x;i--){
            dp[i]+=dp[i-x];
        }
    }
    else{
        // We need to update the value of dp[i] as dp[i]-=dp[i-x] where dp[i-x] is its new value (the updated one)
        for(int i=x;i<=k;i++){
            dp[i]-=dp[i-x];
        }
    }
    cout<<dp[k]<<'\n';
}
```
- DP on trees problems usually require us to consider the tree as a rooted tree and then do DFS while maintaining DP vector(s)
- Quite often, we have to calculate answer for the tree considering all the nodes as root of the tree. This can be done by calculating the answer for one particular root and then using those calculated values to calculate answer for new root (the adjacent node to the original root is taken as new root). For example, consider this problem: There's a tree with $n$ vertices numbered $1$ to $n$. An integer $a_i$ is written on vertex $i$ for $i = 1,2,...,n$. You have to make all $a_i$ equal by performing some (possibly zero) operations. Suppose, you root the tree at some vertex. In each operation, you select any vertex $v$ and a non negative intger $c$. Then for all vertices $i$ in $v$'s subtree, replace $a_i$ with $a_i \oplus c$. The cost of this operation is $s.c$, where $s$ is number of vertices in the subtree. Let $m_r$ denote minimum possible total cost required to make all $a_i$ equal, if vertex $r$ is chosen as root of the tree. Find $m_1, m_2, ..., m_n$
```cpp
// Let's first calculate m1. We do this by making every node's value equal to a[1]. Infact, minimum total cost when calculating mr would be when we make each node's value equal to a[r] (We can prove this requires minimum cost)

void dfs(int u,int p,vector<vector<int>>& g,vector<int>& dp,vector<int>& sz,vector<int>& a){
    sz[u]=1;
    for(auto v:g[u]){
        if(v!=p){
            dfs(v,u,g,dp,sz,a);
            sz[u]+=sz[v];
            dp[u]+=(a[u]^a[v])*sz[v]+dp[v];
        }
    }
}

void dfs2(int u,int p,vector<vector<int>>& g,vector<int>& ans,vector<int>& sz,vector<int>& a){
    for(auto v:g[u]){
        if(v!=p){ // Here, we are changing root of the tree from u to v
            int su=sz[u],sv=sz[v];
            // Updating sz values as per new root
            sz[v]=su; // Updating sz[v]
            sz[u]-=sv; // Updating sz[u]
            ans[v]=ans[u]+(a[u]*a[v])*(su-2*sv);
            dfs2(v,u,g,ans,sz,a);
            // Restoring original sz values (as per old root)
            sz[u]=su;
            sz[v]=sv;
        }
    }
}

void solve(){
    vector<ll>dp(n+1,0),sz(n+1,0);
    // dp[i] -> Minimum total cost to make all vertices in i's subtree equal to a[i] if the tree is rooted at vertex 1
    // sz[i] -> Number of nodes in i's subtree
    dfs(1,0,g,dp,sz,a);
    vector<int>ans(n+1,0);
    ans[1]=dp[1];
    // Now, we have to calculate answer when other nodes are taken as root. We have the following recurrence relation:
    // Suppose u and v are adjacent nodes and we know ans[u]. Then, ans[v] = ans[u] + (a[u] ^ a[v]) * (sz[u] - 2 * sz[v]). Note that the sz values are according to the rooted tree with u as root
    dfs2(1,0,g,ans,sz,a);
    for(int i=1;i<=n;i++){
        cout<<ans[i]<<" ";
    }
}
```
- We can also do DP on DAG (Directed Acyclic Graph). We process the nodes in topological order or reverse topological order
- In Digit DP, we have to answer queries such as "The count of numbers that satisfy property $X$ in $[a, b]$". This can be done by introducing a function $f$ as $f(n) =$ count of numbers $\leq n$ that satisfy property $X$. So answer $= f(b) - f(a - 1)$
- We can use maps instead of arrays for memoizing when values have a large spread
- State reduction techniques are often used to cut down on memory used. As an example, consider the multidimensional Knapsack problem, where we have to maximize energy with constraints on volume and mass. The following code reduces the number of states from $3$ to $2$
```cpp
int max_energy(int V,int M,vector<int>& v,vector<int>& m,vector<int>& e){
    int n=v.size();
    vector<vector<int>>dp1(V+1,vector<int>(M+1,0));
    vector<vector<int>>dp2(V+1,vector<int>(M+1,0));
    bool flag=true;
    for(int i=v[0];i<=V;i++){
        for(int j=m[0];j<=M;j++){
            dp1[i][j]=e[0];
        }
    }
    for(int i=1;i<n;i++){
        for(int j=0;j<=V;j++){
            for(int k=0;k<=M;k++){
                if(flag){
                    dp2[j][k]=dp1[j][k];
                    if(j>=v[i] && k>=m[i]){
                        dp2[j][k]=max(dp2[j][k],e[i]+dp1[j-v[i]][k-m[i]]);
                    }
                }
                else{
                    dp1[j][k]=dp2[j][k];
                    if(j>=v[i] && k>=m[i]){
                        dp1[j][k]=max(dp1[j][k],e[i]+dp2[j-v[i]][k-m[i]]);
                    }
                }
            }
        }
        flag=(flag?false:true);
    }
    if(flag){
        return dp1[V][M];
    }
    return dp2[V][M];
}
```
- Knapsack DP using a single 1D DP array optimization
```cpp
vector<int>dp(W+1,0);
for(int i=0;i<n;i++){
    for(int j=W;j>=0;j--){
        if(w[i]<=j){
            dp[j]=max(dp[j],v[i]+dp[j-w[i]]);
        }
    }
}
// dp[x] represents the maximum value if the knapsack can carry a weight of at most x
```
- We need not visit all the states of DP
- Many Expectations problems require DP for solving
- DP with bitmasking is often used in problems involving recurrence relations on subsets of a set. For example, consider this problem where there are $n$ people ($n \leq 10$), each having a collection of caps ($1$ of each type and each type varies between $1$ and $c$). The collection of caps possessed by each person is given. You need to calculate the number of ways these $n$ people can wear caps such that no two people wear same type of cap
```cpp
vector<int>cap(c+1);
// cap[i] -> Mask which have cap i in their possession
vector<int>dp((1<<n),vector<int>(c+2,0));
// dp[mask][i] -> Number of ways we can assign caps to mask such that no two caps are of same type and all cap types are >= i
for(int i=c+1;i>0;i--){
    for(int mask=0;mask<(1<<n);mask++){
        if(!mask){
            dp[mask][i]=1;
            continue;
        }
        dp[mask][i]=dp[mask][i+1];
        for(int j=0;j<n;j++){
            if((mask&(1<<j)) && (cap[i]&(1<<j)){
                dp[mask][i]+=dp[mask^(1<<j)][i+1];
            }
        }
    }
    cout<<dp[(1<<n)-1][1]<<'\n';
}
```

**6. Combinatorics**

- **Pigeonhole Principle**
- **Stars and Bars method (To determine number of non-negative integer solutions)**: Let the equation be $\Sigma_{i=1}^{r}x_i = n$, where each $x_i \geq 0$. We need to find number of distinct solutions to the given equation. This problem can be modelled as follows. Suppose $n$ identical stars are kept in a straight line. Now, we need to place $(r - 1)$ identical bars to create $r$ partitions. The number of stars to the left of leftmost bar $=$ value of $x_1$. Number of stars to the right of rightmost bar $=$ value of $x_r$. Number of stars between $(i-1)^{th}$ and $i^{th}$ bar (assuming $1$-indexing) $=$ value of $x_i$. Thus the given problem now reduces to finding number of ways to arrange $n$ identical stars and $(r - 1)$ identical bars, and thus equal to $\binom{n+r-1}{n}$
- **Counting number of permutations of a particular string**: Suppose our string contains the alphabets $x_1, x_2, ..., x_n$. Suppose the $i^{th}$ alphabet $x_i$ appears $r_i$ times in the string. Now, number of distinct permutations of the given string is equal to $\binom{r_1+r_2+...+r_n}{r_1} \times \binom{r_2+r_3+...+r_n}{r_2} \times ... \times \binom{r_n}{r_n} = \frac{(r_1+r_2+...+r_n)!}{r_1!r_2!...r_n!}$
- **Coefficient of $x^r$ in $(1-x)^{-n} =$** Coefficient of $x^r$ in $((1-x)^{-1})^{n} =$ Coefficient of $x^r$ in $(1+x+x^{2}+...)^{n} =$ Number of integer solutions of $y_{1}+y_{2}+...+y_{n} = r$ where each $y_{i} \geq 0$ is equal to $\binom{n+r-1}{r}$
- **Stirling Numbers**
    - **Unsigned Stirling number of the first kind**: Denoted by $S_{1}(n,k)$, it is defined as number of permutations of $n$ elements with $k$ cycles. Suppose we have $(n+1)$ elements and we have to partition it into $k$ cycles. Now, the $(n+1)^{th}$ element can be introduced at any of the $n$ places or it can form a separate cycle. Thus recurrence relation for $S_{1}(n+1,k)$ is $S_{1}(n+1,k) = nS_{1}(n,k) + S_{1}(n,k-1)$ for $k > 0$. The base case would be $S_{1}(0,0) = 1$ and $S_{1}(n,0) = S_{1}(0,n) = 0$ for $n > 0$
    - **Stirling number of the second kind**: Denoted by $S(n,k)$, it counts the number of ways to partition a set of $n$ labeled objects into $k$ non-empty unlabeled subsets. Suppose we have $(n+1)$ labeled objects and we have to partition it into $k$ non-empty subsets. Now, the $(n+1)^{th}$ object can be introduced into one of the $k$ existing subsets or it can form a separate subset. Thus recurrence relation for $S(n+1,k)$ is $S(n+1,k) = kS(n,k) + S(n,k-1)$ for $0 < k < n$. The base case would be $S(n,n) = 1$ for $n \geq 0$ and $S(n,0) = S(0,n) = 0$ for $n > 0$
- **Burnside's Lemma (Polya's Enumeration Principle)**
- **Catalan Numbers** ($C_{n} = \frac{\binom{2n}{n}}{n+1}$)
- **Inclusion-Exclusion Principle**: The principle of inclusion-exclusion states that for finie sets $A_{1}, A_{2},...,A_{n}$, one has the identity
  $$\bigg |\bigcup_{i=1}^{n} A_{i}\bigg | = \sum_{i=1}^{n} |A_{i}| - \sum_{1 \leq i < j \leq n} |A_{i} \cap A_{j}| + \sum_{1 \leq i < j < k \leq n} |A_{i} \cap A_{j} \cap A_{k}| - ... + (-1)^{n+1}|A_{1} \cap A_{2} ... \cap A_{n}|$$
- **Binomial Theorem**: It states that $(a+b)^n = \Sigma_{r=0}^{n}\binom{n}{r}a^{n-r}b^{r}$ where $a, b \in \mathbb{R}$ and $n \in \mathbb{N}$
- **Generating Functions**
- **Partitions**
- A nice problem and approach to solve it\
**Problem**: Given non-negative integers $n$ and $m$, find number of non-decreasing sequences of length $n$ such that each element in the sequence lies between $0$ and $m$ inclusive\
\
**Solution**: First, we choose $k$ distinct numbers out of $(m + 1)$ numbers. Then we need to arrange these $k$ numbers into $n$ boxes, so number of ways of arranging is equal to number of integer solutions of $x_1 + x_2 + ... + x_k = n$, where each $x_i \geq 1$. So required answer $= \Sigma_{k=1}^{n}\binom{m+1}{k}\binom{n-1}{k-1} = \Sigma_{k=1}^{n}\binom{m+1}{k}\binom{n-1}{n-k} =$ Coefficient of $x^n$ in $(1+x)^{m+1} \cdot (1+x)^{n-1} =$ Coefficient of $x^n$ in $(1+x)^{m+n} = \binom{m+n}{n}$
- Another nice problem\
  **Problem**: Let $n > 0$ be an integer. We are given a balance and $n$ weights of weight $2^{0}, 2^{1}, ..., 2^{n-1}$. We are to place each of the $n$ weights on the balance, one after another, in such a way that the right pan is never heavier than the left pan. At each step, we choose one of the weights that has not yet been placed on the balance and place it on either the left pan or the right pan, until all of the weights have been placed. Determine the number of ways in which this can be done\
\
**Solution 1**: Let our answer be $W(n)$. It is evident that $W(1) = 1$. Now, since the weights are powers of $2$, it is easy to see that a heavier weight is heavier than all the lighter weights combined. Thus, we only need to ensure that at any point in time, the maximum weight on the left pan is $>$ the maximum weight on the right pan. Thus, the heaviest weight must be placed on the left pan. Suppose the heaviest weight is placed on the left pan at position $k$. Now, the first $(k-1)$ weights can be chosen in $\binom{n-1}{k-1}$ ways and these weights should be placed in $W(k-1)$ ways. The rest of the $(n-k)$ weights can be placed in $2^{n-k}(n-k)!$ ways. Thus the recurrence relation for $W(n)$ is as follows
$$W(n) = \sum_{k=1}^{n}\binom{n-1}{k-1}W(k-1)2^{n-k}(n-k)! = \sum_{k=1}^{n}W(k-1)2^{n-k}\frac{(n-1)!}{(k-1)!} = \sum_{k=1}^{n-1}W(k-1)2^{n-k}\frac{(n-1)!}{(k-1)!} + W(n-1)$$
The above can be simplified as
$$W(n) = \sum_{k=1}^{n-1}W(k-1)2^{n-k}\frac{(n-1)!}{(k-1)!} + W(n-1) = 2(n-1)\sum_{k=1}^{n-1}W(k-1)2^{n-k-1}\frac{(n-2)!}{(k-1)!} + W(n-1) = 2(n-1)W(n-1) + W(n-1) = (2n-1)W(n-1)$$
Thus, we have $W(n) = (2n-1)W(n-1) = (2n-1)(2n-3)W(n-2) = ... = (2n-1)(2n-3)...(3)W(1) = 1\times3\times5...\times(2n-1)$ as $W(1) = 1$\
\
**Solution 2**: Again, let our answer be $W(n)$. We can use recursion on last weight placement. There are $(2n-1)$ choices; namely, you can have any of the $n$ weights to be placed at last and each could be placed in either of the pans, except for the case where the last weight is the heaviest one and it is placed in the right pan. Thus, we get the same recurrence relation as above, that is $W(n) = (2n-1)W(n-1)$ with base case $W(1) = 1$
- There is a common combinatorial trick for counting: We change the perspective to count. For example, suppose we have to count the number of good objects of type $A$ each object of type $B$ yields. Another way to count this is as follows: We count how many objects of type $B$ yield each of the possible good objects of type $A$. So basically, the code changes as follows (Let's call this contribution technique)
```cpp
/* Older version */
int count = 0
for every type B object:
    count += number of good type A objects yielded by it

/* Newer version */
int count = 0
for every good type A object:
    count += number of type B objects that yield it
```
- Computing $\binom{n}{r}$ modulo $MOD$
```cpp
vector<vector<ll>>C(N+1,vector<ll>(N+1,0));
for(int i=1;i<=N;i++){
    for(int j=0;j<=i;j++){
        if(!j || i==j){
            C[i][j]=1;
        }
        else{
            C[i][j]=(C[i-1][j-1]+C[i-1][j])%MOD;
        }
    }
}

// Another way is to calculate the factorials values and then use the formula for nCr
C[n][r]=((factorial[n]*inverse_factorial[r])%MOD*inverse_factorial[n-r])%MOD;
```
- Below code finds count of numbers $\leq m$ which are divisible by atleast one of the numbers in $factors$ by using the inclusion-exclusion principle
```cpp
void pie(int idx,vector<int>& factors,int n,int m,int product,int c,int& cnt){
    if(product>m){
        return;
    }
    if(idx==n){
        if(!c){
            return;
        }
        if(c&1){
            cnt+=m/product;
        }
        else{
            cnt-=m/product;
        }
        return;
    }
    pie(idx+1,factors,n,m,product,c,cnt);
    pie(idx+1,factors,n,m,product*factors[idx],c+1,cnt);
}

void solve(){
    int m,cnt=0;
    vector<int>factors;
    int n=factors.size();
    int product=1; // Product of numbers in the subset of "factors"
    int c=0 // Count of numbers in subset of "factors"
    pie(0,factors,n,m,product,c,cnt);
    cout<<cnt;
}
```
- Matrix exponentiation can be used to calculate $n^{th}$ term of a linear recurrence in $O(m^3$ $log$ $n)$ time where $m$ denotes size of the matrix
- Many counting problems can be solved by fixing some parameters and iterating on them. Also, many counting problems can be solved by standard techniques such as fixing the $L$ pointer or $2$ pointer method
- Many combinatorial problems require DP for solving
- Some useful identities
    - $\binom{n}{r} + \binom{n}{r-1} = \binom{n+1}{r}$ (Suppose there are $n$ normal objects and $1$ special object. Now, $RHS =$ Number of ways of choosing $r$ objects out of these $n+1$ objects $=$ Number of ways of choosing $r$ objects by excluding the special object $+$ Number of ways of choosing $r$ objects by including the special object = $\binom{n}{r} + \binom{n}{r-1} = LHS$)
    - $\Sigma_{i=r}^{n}\binom{i}{r} = \binom{n+1}{r+1}$ $(LHS =$ Coefficient of $x^r$ in $\Sigma_{i=r}^{n}(1+x)^i =$ Coefficient of $x^r$ in $\(1+x)^r\frac{(1+x)^{n-r+1}-1}{(1+x)-1}$ = Coefficient of $x^r$ in $\frac{(1+x)^{n+1}-(1+x)^r}{x}$ = Coefficient of $x^{r+1}$ in $(1+x)^{n+1}-(1+x)^r = \binom{n+1}{r+1} - 0 = RHS$)
    - $\Sigma_{k=0}^{r}\binom{m}{k}\binom{n}{r-k} = \binom{m+n}{r}$ $(LHS =$ Coefficient of $x^r$ in $(1+x)^m\cdot(1+x)^n$ = Coefficient of $x^r$ in $(1+x)^{m+n}$ = $\binom{m+n}{r} = RHS)$

**7. Number Theory**

- **Fermat's Little Theorem**
- **Euler's Totient Function**
- **Chinese Remainder Theorem**
- **Matrix Exponentiation**
  
**7.1. Sieve of Eratosthenes and Prime Factorization**

``` cpp
vector<int>mind(N+1); // mind[i] -> minimum prime divisor of i
for(int i=0;i<=N;i++){
    mind[i]=i;
}
for(int i=2;i*i<=N;i++){
    if(mind[i]==i){
        for(int j=i*i;j<=N;j+=i){
            mind[j]=min(mind[j],i);
        }
    }
}
```
- Now after storing minimum prime divisors of every number till $N$, we can do prime factorization easily as follows
```cpp
vector<pair<int,int>>prime_factors
// Each element of prime_factors is of the form (prime factor, its power)
while(N>1){
    int p=mind[N],cnt=0;
    while(!(N%p)){
        N/=p;
        cnt++;
    }
    prime_factors.push_back(make_pair(p,cnt));
}
```
- Sometimes $N$ is very high so for determining its prime factors, we can find its prime factors till $\sqrt{N}$ and correspondingly divide $x$ ($=N$ originally) by its prime factors. At last, if $x>1$, then that particular value of $x$ is the last prime factor of $N$. Consider the code below for reference for finding prime factors of a number $N$ where $1 \leq N \leq 10^9$
```cpp
vector<bool>is_prime(1e5,true);
vector<int>primes;

void sieve(){
    is_prime[0]=false;
    is_prime[1]=false;
    for(int i=2;i*i<=1e5;i++){
        if(is_prime[i]){
            for(int j=i*i;j<=1e5;j+=i){
                prime[j]=false;
            }
        }
    }
}

vector<int> find_prime_factors(int N){
    vector<int>prime_factors;
    int x=N;
    int k=primes.size();
    for(int i=0;i<k;i++){
        if(x==1){
            break;
        }
        if(!(x%primes[i])){
            prime_factors.push_back(primes[i]);
            while(!(x%primes[i])){
                x/=primes[i];
            }
        }
    }
    if(x>1){
        prime_factors.push_back(x);
    }
    return prime_factors;
}

void solve(){
    sieve();
    for(int i=2;i<=1e5;i++){
        if(is_prime[i]){
            primes.push_back(i);
        }
    }
    int N=58495940;
    vector<int>ans = find_prime_factors(N);
}
```

**7.2. Greatest Common Divisor (GCD)**

```cpp
int gcd(int a,int b){
    if(!a){
        return b;
    }
    return gcd(b%a,a);
}
```

**7.3. Binary Exponentiation**

```cpp
int bin_exp(int x,int n){
    int res=1;
    while(n){
        if(n&1){
            res*=x;
        }
        n>>=1;
        x*=x;
    }
    return res;
}
```
- Above can be modified to make the operation modulo $m$

**7.4. Primality test**

```cpp
bool is_prime(int x){
    for(int d=2;d*d<=x;d++){
        if(!(x%d)){
            return false;
        }
    }
    return true;
}
```

**7.5. Storing factors of a number**

```cpp
vector<int>factors;
for(int i=1;i*i<=N;i++){
    if(i*i==N){
        factors.push_back(i);
        break;
    }
    if(!(N%i)){
        factors.push_back(i);
        factors.push_back(N/i);
    }
}
```

**7.6. Modular inverse**

- Calculating modular inverse when $MOD$ is a prime is straightforward. We can use Fermat's Little Theorem. Suppose we have to find inverse of $a$ modulo $MOD$ where $MOD$ is prime. Then $a^{-1} = a^{MOD-2}$ $modulo$ $MOD$. If $MOD$ is not prime, then we have to use Extended Euclidean Algorithm to find the multiplicative inverse. Note that the inverse of $a$ modulo $MOD$ exists iff $gcd(a, MOD) = 1$

**7.7. Some general notes and ideas**

- Suppose we are given an array of $n$ integers $[a_1, a_2, ..., a_n]$ and a prime number $p$. Suppose we need to check whether for every integer $x > 0$, $p$ divides atleast $2$ of $[a_1+x, a_2+x, ..., a_n+x]$. To check this, we reduce our array to $[a_1$ $mod$ $p, a_2$ $mod$ $p, ..., a_n$ $mod$ $p]$. Now, if $min(cnt_0, cnt_1, ..., cnt_{p-1}) > 1$ in our reduced array, then $p$ divides atleast $2$ of $[a_1+x, a_2+x, ..., a_n+x]$ $\forall x > 0$. Else there exists some $x$ such that $p$ divides atmost $1$ of $[a_1+x, a_2+x, ..., a_n+x]$
- The Harmonic series $\Sigma_{i=1}^{n} \frac{1}{i} = O(log$ $n)$ is very useful in determining time complexities of many algorithms
- There are $O(\sqrt{n})$ distinct values of $\lfloor\frac{n}{i}\rfloor$ when $i$ varies from $1$ to $n$
- The maximum value of $j$ such that $\lfloor\frac{n}{i}\rfloor = \lfloor\frac{n}{j}\rfloor =$ $x$ (say) is given by $j = \lfloor\frac{n}{x}\rfloor$
- $gcd(kp, kq) = k \cdot gcd(p, q)$ and $lcm(kp, kq) = k \cdot lcm(p,q)$
- Suppose prime factorization of a number $N = p_1^{\alpha_1}p_2^{\alpha_2}...p_k^{\alpha_k}$
    - Number of factors of $N = (1+\alpha_1)(1+\alpha_2)...(1+\alpha_k)$
    - Sum of factors of $N = (\frac{p_1^{\alpha_1+1}-1}{p_1-1})(\frac{p_2^{\alpha_2+1}-1}{p_2-1})...(\frac{p_k^{\alpha_k+1}-1}{p_k-1})$
    - Product of factors of $N = \sqrt{N^{\textrm{Number of factors of N}}}$
    - $\varphi(N) = N(1-\frac{1}{p_1})(1-\frac{1}{p_2})...(1-\frac{1}{p_k})$
- Bezout's Identity: Let $a$ and $b$ be integers with $gcd$ $d$. Then there exist integers $x$ and $y$ such that $ax + by = d$. Moreover, the integers of the form $az + bt$ are exactly the multiples of $d$ ($d$ is the smallest positive integer in the set $S = [ ax + by : x,y \in \mathbb{Z} \textrm{ and } ax + by > 0 ]$)
- Chicken McNugget Theorem: For any two relatively prime positive integers $m$ and $n$, the greatest integer that cannot be written in the form $am + bn$ for non-negative integers $a, b$ is $mn - m - n$
- Suppose $n$ events occur after every $t_{1}, t_{2}, ..., t_{n}$ seconds and suppose, all these events occur together at $t = 0$. These events will occur again after every $lcm(t_{1}, t_{2}, ..., t_{n})$ seconds
- Suppose prime factorization of numbers $N_{1}$ and $N_{2}$ is as follows: $N_{1} = p_1^{\alpha_1}p_2^{\alpha_2}...p_k^{\alpha_k}$ and $N_{2} = p_1^{\beta_1}p_2^{\beta_2}...p_k^{\beta_k}$. Then
    - $gcd(N_{1}, N_{2}) =  p_1^{min(\alpha_1,\beta_1)}p_2^{min(\alpha_2,\beta_2)}...p_k^{min(\alpha_k,\beta_k)}$
    - $lcm(N_{1}, N_{2}) =  p_1^{max(\alpha_1,\beta_1)}p_2^{max(\alpha_2,\beta_2)}...p_k^{max(\alpha_k,\beta_k)}$
- Suppose $p$ is a prime number and $n \in \mathbb{N}$. The highest power of $p$ which divides $n!$ is given by $\Sigma_{i=1}^{\infty}\lfloor\frac{n}{p^i}\rfloor$

**8. Graph Theory**

**8.1. Depth First Search (DFS)**

```cpp
void dfs(int u,vector<vector<int>>& graph,vector<int>& color){
    color[u]=1;
    for(auto v:graph[u]){
        if(!color[v]){
            dfs(v,graph,color);
        }
    }
    color[u]=2;
}
```
- Often used while doing DP on Trees
- DFS Tree can be used to solve many problems

**8.2. Breadth First Search (BFS)**

```cpp
queue<int>q;
vector<bool>visited(n,false);
q.push(0);
visited[0]=true;
while(!q.empty()){
    int u=q.front();
    q.pop();
    for(auto v:graph[u]){
        if(!visited[v]){
            q.push(v);
            visited[v]=true;
        }
    }
}
```
- Multi-Source BFS can be used to find distance of nearest node among a set of nodes. Just put all the nodes of the set in the queue instead of just $1$ of them
  
**8.3. Dijkstra's Algorithm**

```cpp
// Implementation using set
void dijkstra(int source,vector<int>& distance,vector<int>& parent,vector<vector<pair<int,int>>>& graph,int nodes){
    distance.assign(nodes,1e18);
    parent.assign(nodes,-1);
    distance[source]=0;
    set<pair<int,int>>st;
    st.insert({0,source});
    while(!st.empty()){
        int v=st.begin()->second;
        st.erase(st.begin());
        for(auto x:graph[v]){
            if(distance[v]+x.second<distance[x.first]){
                st.erase({distance[x.first],x.first});
                distance[x.first]=distance[v]+x.second;
                parent[x.first]=v;
                st.insert({distance[x.first],x.first});
            }
        }
    }
}
```
```cpp
// Implementation using priority queue
void dijkstra(int source,vector<int>& distance,vector<int>& parent,vector<vector<pair<int,int>>>& graph,int nodes){
    distance.assign(nodes,1e18);
    parent.assign(nodes,-1);
    distance[source]=0;
    priority_queue<pair<int,int>,vector<pair<int,int>>,greater<pair<int,int>>>pq;
    pq.push({0,source});
    while(!pq.empty()){
        int v=pq.top().second();
        int d_v=pq.top().first;
        pq.pop();
        if(d_v!=distance[v]){
            continue;
        }
        for(auto x:graph[v]){
            if(distance[v]+x.second<distance[x.first]){
                distance[x.first]=distance[v]+x.second;
                parent[x.first]=v;
                pq.push({distance[x.first],x.first});
            }
        }
    }
}
```
- Multi-Source Dijkstra can be used to find shortest distance from a set of nodes. Just assign $0$ distance to all these source nodes and push them in set/priority queue

**8.4. Kruskal's Algorithm**

```cpp
// edges[i][0] -> weight of ith edge
// edges[i][1] -> 1st node of ith edges
// edges[i][2] -> 2nd node of ith edge
// get() and unite() are standard DSU functions
int kruskal(vector<vector<int>>& edges){
    sort(edges.begin(),edges.end());
    int weight=0;
    for(int i=0;i<n;i++){
        int u=get(edges[i][1],p);
        int v=get(edges[i][2],p);
        if(u!=v){
            unite(u,v,p,r);
            weight+=edges[i][0];
        }
    }
    return weight;
}
```

**8.5. Bellman Ford Algorithm**

**8.6. Floyd Warshall Algorithm**

**8.7. Topological Sort**

- Can be done by doing DFS and storing the visited vertices (whose DFS tree is generated) in a stack
- A topological ordering is unique iff for every two adjacent nodes in the ordering, an edge exists between these two nodes in the graph. Another method to check uniqueness is that length of longest path in the graph = number of vertices $-$ $1$

**8.8. Some general notes and ideas**

- Usually, graphs are constructed using $2D$ vectors. But maps can also be used for constructing graphs
- A graph is bipartite iff it contains no odd cycles
- A graph is a cycle graph iff each vertex has degree $2$ and number of connected components is $1$
- Many problems can be solved by modelling the problem into a known graph problem and then apply known algorithms on this graph like BFS, DFS, Dijkstra, Topological Sort, etc
- To calculate farthest node (and its distance) for each node in a tree, we can do the following. Pick an arbitrary vertex (let's say $u$) and find vertex farthest from $u$ (let's say it is $v_1$). Now from $v_1$, find farthest vertex (let's say $v_2$). Note that $v_1$ and $v_2$ are ends of the diameter of the tree. For each node in the tree, the farthest node is one of the ends of the diameter of the tree (i.e. $v_1$ or $v_2$). Thus we can calculate distances of each node from $v_1$ and $v_2$ and then take maximum among those $2$ distances to get distance from farthest node
- In many problems, the idea is to make a graph and find shortest distance between $2$ nodes. But building such a graph can be very time-consuming and can cause TLE if we build such a graph. Instead, the idea is to build a bipartite graph and find shortest distance on this graph. For example, consider a graph that contains an edge between two nodes iff they are non-coprime. We can instead build a bipartite graph where the left set contains array elements and right set contains prime numbers. A node $u$ in left part is connected to a node $p$ in right part iff $p|u$. Then, we can find shortest distance between two nodes $u$ and $v$ in this bipartite graph and divide the answer by $2$. Another example is a graph where two nodes are connected iff both the sets share a common element. In this case, build the bipartite graph such that left part consists of set number and right part consists of set elements. A node $u$ in left part is connected to a node $e$ in right part iff $e \in u$
- Many problems involve finding shortest distance from a particular node with some minor tweaks, for example reversing the edges of the graph atmost once. In such problems, we have to introduce some dummy nodes and add edges of appropriate weight and then run Dijkstra's Algorithm on it

**9. Bit Manipulation**

- Bit manipulation problems often require constructing the answer bit by bit/adding the contribution of each bit to the answer
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
- search() can be modified to return number of matching characters
- Can be used to solve problems related to finding maximum/minimum xor/xnor of two integers in an array

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

- Can support all range queries where it is possible to divide a range into two parts, calculate the answer separately for both parts and then efficiently combine the answers. Example of such queries are minimum and maximum, greatest common divisor, and bit operations and, or and xor
- Consider this template for calculating sums in an array using segment tree
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
/* Let us assume that we have processed till index i in the sorted array and we can write every integer from 1 to sum[1..i] (sum till index i) as a sum of subset of x[1..i]. Now, if x[i+1] <= sum[1..i]+1, then we can write every number from sum[1..i]+1 to sum[1..i]+x[i+1] by using x[i+1] and writing the remaining sum from subset of x[1..i]. Also, we can write any number from 1 to sum[1..i] by using subset of x[1..i] (By assumption). Thus, we can write all integers from 1 to sum[1..(i+1)] as sum of elements of subset of x[1..(i+1)]. Thus, by induction, all integers from 1 to sum[1..i] can be written as sum of elements of subset of x[1..i], if x[i]<=sum[1..(i-1)]+1. If x[i]>sum[1..(i-1)]+1, then we can never write the number p = sum[1..(i-1)]+1 as sum of elements of subset of x[1..i]. Thus answer would be p */
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
- Consider the problem of finding the longest subarray and number of subarrays such that difference between maximum and minimum elements in the subarray doesn't exceed $k$. This can be easily solved using two pointer method
```cpp
// Assuming array elements in a[0...(n-1)]
multiset<int>mst;
int l=0,longest=0,count=0;
for(int r=0;r<n;r++){
    mst.insert(a[r]);
    while(l<r){
        int mx=*(--mst.end());
        int mn=*mst.begin();
        if(mx-mn<=k){
            break;
        }
        mst.erase(mst.find(a[l]));
        l++;
    }
    longest=max(longest,r-l+1);
    count+=r-l+1;
}
// longest -> Longest subarray with difference between maximum and minimum elements <= k
// count -> Number of subarrays with difference between maximum and minimum elements <= k
```
Above solution works in $O(n$ $log$ $n)$ time. We can reduce the time complexity to $O(n)$ by using two stacks as follows
```cpp
stack<int>f,b,mxf,mxb,mnf,mnb; // f stands for forward stack and b stands for backward stack. These stacks are used to simulate queue

void add(int x){
    f.push(x);
    int mx=x,mn=x;
    if(!mxf.empty()){
        mx=max(mx,mxf.top());
    }
    if(!mnf.empty()){
        mn=min(mn,mnf.top());
    }
    mxf.push(mx);
    mnf.push(mn);
}

void remove(){
    if(b.empty()){
        while(!f.empty()){
            b.push(f.top());
            if(mxb.empty()){
                mxb.push(f.top());
            }
            else{
                mxb.push(max(mxb.top(),f.top()));
            }
            if(mnb.empty()){
                mnb.push(f.top());
            }
            else{
                mnb.push(min(mnb.top(),f.top()));
            }
            f.pop();
            mxf.pop();
            mnf.pop();
        }
    }
    b.pop();
    mxb.pop();
    mnb.pop();
}

void solve(){
    // Assuming array elements in a[0...(n-1)]
    int l=0,longest=0,count=0;
    for(int r=0;r<n;r++){
        add(a[r]);
        while(l<r){
            int mx=INT_MIN,mn=INT_MAX;
            if(!mxf.empty()){
                mx=max(mx,mxf.top());
            }
            if(!mxb.empty()){
                mx=max(mx,mxb.top());
            }
            if(!mnf.empty()){
                mn=min(mn,mnf.top());
            }
            if(!mnb.empty()){
                mn=min(mn,mnb.top());
            }
            if(mx-mn<=k){
                break;
            }
            remove();
            l++;
        }
        longest=max(longest,r-l+1);
        count+=r-l+1;
    } 
    // longest -> Longest subarray with difference between maximum and minimum elements <= k
    // count -> Number of subarrays with difference between maximum and minimum elements <= k
}
```
Another problem illustrating the usefulness of using two stacks: Given an array of $n$ integers $a_i$. A segment on this array $a[l...r]$ is good if $GCD$ of all numbers in this segment is $1$. Find length of shortest such segment or print $-1$ if no good segment exists
```cpp
stack<int>f,b,fgcd,bgcd; // f stands for forward stack and b stands for backward stack. These stacks are used to simulate queue

void add(int x){
    f.push(x);
    if(!fgcd.empty()){
        fgcd.push(__gcd(fgcd.top(),x));
    }
    else{
        fgcd.push(x);
    }
}

void remove(){
    if(b.empty()){
        while(!f.empty()){
            b.push(f.top());
            if(bgcd.empty()){
                bgcd.push(f.top());
            }
            else{
                bgcd.push(__gcd(bgcd.top(),f.top()));
            }
            f.pop();
            fgcd.pop();
        }
    }
    b.pop();
    bgcd.pop();
}

void solve(){
    // Assuming array integers in a[0...(n-1)]
    int g;
    for(int i=0;i<n;i++){
        if(!i){
            g=a[i];
        }
        else{
            g=__gcd(g,a[i]);
        }
    }
    if(g!=1){
        cout<<"-1\n";
        return;
    }
    int l=0,ans=n;
    for(int r=0;r<n;r++){
        add(a[r]);
        while(l<=r){
            int gcd=INT_MAX;
            if(!fgcd.empty()){
                gcd=fgcd.top();
            }
            if(!bgcd.empty()){
                if(gcd==INT_MAX){
                    gcd=bgcd.top();
                }
                else{
                    gcd=__gcd(gcd,bgcd.top());
                }
            }
            if(gcd==1){
                ans=min(ans,r-l+1);
            }
            else{
                break;
            }
            if(l==r){
                break;
            }
            remove();
            l++;
        }
    }
    cout<<ans<<'\n';
}
```
Another problem illustrating use of bitsets and the two stacks technique: Given an array of $n$ integers $a_i$. A segment on this array $a[l...r]$ is good if it is possible to choose a certain set of numbers whose sum is equal to $s (1 \leq s \leq 1000)$. Find shortest such segment or print $-1$ if no such segment exists
```cpp
stack<int>f,b; // f stands for forward stack and b stands for backward stack. These stacks are used to simulate queue
stack<bitset<1001>>fb,bb;

void add(int x){
    f.push(x);
    if(fb.empty()){
        bitset<1001>bt;
        bt[0]=1;
        bt[x]=1;
        fb.push(bt);
    }
    else{
        bitset<1001>bt=fb.top();
        bt=bt|(bt<<x);
        fb.push(bt);
    }
}

void remove(){
    if(b.empty()){
        while(!f.empty()){
            b.push(f.top());
            if(bb.empty()){
                bitset<1001>bt;
                bt[0]=1;
                bt[f.top()]=1;
                bb.push(bt);
            }
            else{
                bitset<1001>bt=bb.top();
                bt=bt|(bt<<f.top());
                bb.push(bt);
            }
            f.pop();
            fb.pop();
        }
    }
    b.pop();
    bb.pop();
}

void solve(){
    // Assuming array elements in a[0...(n-1)]
    int ans=INT_MAX,l=0;
    for(int r=0;r<n;r++){
        add(a[r]);
        while(l<=r){
            bitset<1001>bt1,bt2;
            if(!fb.empty()){
                bt1=fb.top();
            }
            if(!bb.empty()){
                bt2=bb.top();
            }
            bool f=0;
            if(bt1[s]==1 || bt2[s]==1){
                ans=min(ans,r-l+1);
                f=1;
            }
            for(int i=0;i<=s;i++){
                if(bt1[i]==1 && bt2[s-i]==1){
                    ans=min(ans,r-l+1);
                    f=1;
                    break;
                }
            }
            if(l==r || !f){
                break;
            }
            remove();
            l++;
        }
    }
    if(ans==INT_MAX){
        ans=-1;
    }
    cout<<ans<<'\n';
}
```
