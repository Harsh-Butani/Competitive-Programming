# Notes

## For theory, refer the following links:
- [Codeforces Catalog](https://codeforces.com/catalog)
- [Codeforces EDU](https://codeforces.com/edu/courses)
- [CP-Algorithms](https://cp-algorithms.com/)
- [Topcoder - Competitive Programming](https://www.topcoder.com/thrive/tracks?track=Competitive%20Programming)
- [USACO Guide](https://usaco.guide/)
  
**1. Binary Search**
- Suppose we are given the following:
    - $[l,r]$: a range of integers
    - $f$: a function from integers to booleans which satisfies following property: there exists some integer $t$ such that for all $l \leq x < t$, $f(x)$ is true and for all $t \leq x \leq r$, $f(x)$ is false ($f$ is called a predicate)
- General code looks something like this
```cpp
int lo=l,hi=r,mid,ans1=l-1,ans2=r+1;
while(lo<=hi){
    mid=lo+(hi-lo)/2;
    if(f(mid)){
        ans1=mid;
        lo=mid+1;
    }
    else{
        ans2=mid;
        hi=mid-1;
    }
}
return ans2; // first false
// return ans1 for last true
```
- Another way of implementing binary search: 
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
}
return hi; // first false
// return lo for last true
```
- Binary search to search elements in sorted arrays
- Binary search on answer (often used in finding maximum/minimum value of certain quantities subject to some constraints). Consider this problem. You have been given a set $S$, initially containing all positive integers in sorted order. Everyday, you will remove $a_{1}^{th}, a_{2}^{th}, ..., a_{n}^{th}$ smallest numbers in $S$ simultaneously $(a_{1} < a_{2} < ... < a_{n})$. Find the smallest element in $S$ after $k$ days
```cpp
/*
We shall assume 1-based indexing of array a. We shall binary search on answer. Let the current
candidate for the answer be x. We will determine the minimum number of days to remove x from the
set. We will binary search on minimum x such that minimum number of days to remove it is > k
*/
if(a[1]>1){
    cout<<"1\n";
    return;
}
int lo=1,hi=a[n]+n*k,mid,ans;
while(lo<=hi){
    mid=lo+(hi-lo)/2;
    int pos=mid,days=0;
    for(int i=n;i>0;i--){
        if(pos<a[i]){
            continue;
        }
        int d=(pos-a[i])/i+1;
        pos-=i*d;
        days+=d;
    }
    if(days<=k){
        lo=mid+1;
    }
    else{
        ans=mid;
        hi=mid-1;
    }
}
cout<<ans<<'\n';
```
- Often used in minimax/maximin problems. For example, consider this problem: You are given an array $a$ containing $n$ positive integers. Divide this array into $k$ partitions $(1 \leq k \leq n)$ such that the minimum sum of elements among all $k$ partitions is maximized. Determine this sum
```cpp
/*
We shall assume 1-based indexing of array a. We will binary search on answer. Let the current
candidate for answer be x. By keeping x as the minimum sum, we will try to partition the array
into as many segments as possible. We will binary search on maximum x such that maximum number
of segments in which we can partition the array by keeping x as minimum sum is >= k
*/
int sum=0,mx=0;
for(int i=1;i<=n;i++){
    sum+=a[i];
    mx=max(mx,a[i]);
}
int lo=mx,hi=sum,mid,ans;
while(lo<=hi){
    mid=lo+(hi-lo)/2;
    int p=0,s=0;
    for(int i=1;i<=n;i++){
        s+=a[i];
        if(s>=mid){
            p++;
            s=0;
        }
    }
    if(p>=k){
        ans=mid;
        lo=mid+1;
    }
    else{
        hi=mid-1;
    }
}
cout<<ans<<'\n';
```
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
- Above solution works in $O(n$ $log$ $n)$ time. We can reduce the time complexity to $O(n)$ by using two stacks as follows
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
- A problem illustrating the use of difference arrays: We have a directed tree with $n$ vertices. For each $i = 1, 2, ..., n$, count the number of vertices that can possibly appear at $i^{th}$ position of a topological ordering of this tree
```cpp
/*
Suppose node u appears at i-th position in one of the topological orderings. Then,
node u must have atmost (i-1) ancestors and atmost (n-i) children. Thus count of
nodes that can appear at position i in the ordering = count of nodes that have
number of ancestors < i and number of children <= (n-i). Thus a node u which has
x ancestors and y children can appear at any position in the range [x+1,n-y]
*/

void dfs(int u,int p,vector<vector<int>>& graph,vector<int>& ancestors,vector<int>& children){
    for(auto v:graph[u]){
        if(v==p){ // Though this condition would never occur since graph is directed
            continue;
        }
        ancestors[v]=ancestors[u]+1;
        dfs(v,u,graph,ancestors,children);
        children[u]+=1+children[v];
    }
}

void solve(){
    int root=0;
    for(int i=1;i<=n;i++){
        if(!indegree[i]){
            root=i;
            break;
        }
    }
    vector<int>ancestors(n+1,0),children(n+1,0),ans(n+1,0);
    dfs(root,0,graph,ancestors,children);
    for(int u=1;u<=n;u++){
        int mn=ancestors[u]+1;
        int mx=n-children[u];
        if(mn<=mx){
            ans[mn]++;
            if(mx<n){
                ans[mx+1]--;
            }
        }
    }
    for(int i=1;i<=n;i++){
        ans[i]+=ans[i-1];
        cout<<ans[i]<<" ";
    }
}
```
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
        /*
        For any associative and commutative function, we can do something like this
        sum[x]+=sum[y];
        min[x]=min(min[x],min[y]);
        max[x]=max(max[x],max[y]);
        */
    }
    else{
        p[x]=y;
        /*
        For any associative and commutative function, we can do something like this
        sum[y]+=sum[x];
        min[y]=min(min[x],min[y]);
        max[y]=max(max[x],max[y]);
        */
    }
}
```
- Sometimes, it's better to avoid path compression. A problem illustrating this: There are $n$ players who fight monsters. When a monster is defeated, all the players of the team who defeated the monster get the same amount of experience points. There are three types of queries:
    - *join* $X$ $Y$: join two teams to which players $X$ and $Y$ belong to (do nothing if they belong to same team)
    - *add* $X$ $V$: add $V$ experience points to team to which player $X$ belongs
    - *get* $X$: output the current experience points of player $X$
```cpp
/*
We shall avoid path compression here. When operation 'add' is encountered, we add points
to the representative of the team. When 'join' is encountered, we use normal DSU unite()
and store the difference between the current and its parent node. When 'get' is
encountered, we retrieve the points of its representative node and then travel down the
tree to the requested node while adding the difference we had stored
*/

class DSU{
public:
    int n;
    vector<int>p,r,add,sum;
    
    DSU(int _n){
        n=_n;
        p.resize(n+1,0);
        iota(p.begin(),p.end(),0);
        r.resize(n+1,0);
        add.resize(n+1,0);
        sum.resize(n+1,0);
    }
    
    int find(int x){
        return x==p[x]?x:find(p[x]);
    }
    
    void unite(int u,int v){
        u=find(u);
        v=find(v);
        if(u!=v){
            if(r[u]==r[v]){
                r[u]++;
            }
            if(r[u]>r[v]){
                p[v]=u;
                add[v]=sum[v]-sum[u];
            }
            else{
                p[u]=v;
                add[u]=sum[u]-sum[v];
            }
        }
    }
};

void solve(){
    DSU d(n);
    while(q--){
        string s;
        cin>>s;
        if(s=="join"){
            int x,y;
            cin>>x>>y;
            d.unite(x,y);
        }
        else if(s=="add"){
            int x,v;
            cin>>x>>v;
            d.sum[d.find(x)]+=v;
        }
        else{
            int x;
            cin>>x;
            int ans=0;
            while(x!=d.p[x]){
                ans+=d.add[x];
                x=d.p[x];
            }
            ans+=d.sum[x];
            cout<<ans<<'\n';
        }
    }
}
```
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
- DP on trees problems usually require us to consider the tree as a rooted tree and then do DFS while maintaining DP vector(s). For example, consider this problem: Consider a tree with $n$ vertices rooted at vertex $1$. For some permutation $a$ of $[1...n]$, let $f(a)$ be the number of pairs of vertices $(u,v)$ such that $a_{u} < a_{lca(u,v)} < a_{v}$. Find maximum possible value of $f(a)$ over all permutations of $[1...n]$
```cpp
/*
Note that in order to maximize f(a), we can always assign values to the tree such
that the values in the subtree of vertex x are either all > a[x] or all < a[x].
To maximize the value of f(a), we must assign the values such that the sum of
sizes of subtree(s) with value > a[x] and that of value < a[x] are as close as
possible 
*/

void dfs(int u,int p,vector<vector<int>>& g,vector<int>& dp,vector<int>& sz){
    vector<int>x;
    sz[u]=1;
    for(auto v:g[u]){
        if(v!=p){
            dfs(v,u,g,dp,sz);
            dp[u]+=dp[v];
            sz[u]+=sz[v];
            x.push_back(sz[v]);
        }
    }
    int W=sz[u]/2,n=(int)x.size();
    vector<int>dp2(W+1,0);
    for(int i=0;i<n;i++){
        for(int j=W;j>=0;j--){
            if(j>=x[i]){
                dp2[j]=max(dp2[j],x[i]+dp2[j-x[i]]);
            }
        }
    }
    dp[u]+=dp2[W]*(sz[u]-1-dp2[W]);
}

void solve(){
    vector<int>dp(n+1,0),sz(n+1,0);
    // dp[i] -> Number of pairs (u,v) satisfying the given criteria in i's subtree
    // sz[i] -> Number of nodes in i's subtree
    dfs(1,0,g,dp,sz);
    cout<<dp[1]<<'\n';
}
```
- Some Tree DP problems involve doing DP over the other part of the current subtree as well. This is usually done by passing another parameter in the $DFS()$ function. For example, consider this problem: You have a tree with $n$ vertices, some of which are marked. Let $f_{i}$ denote the maximum distance from vertex $i$ to one of the marked vertices. Find value of $f_{i}$ for all vertices of the tree
```cpp
/*
Note that if we were to find minimum distance from marked vertices for all the vertices,
we could have done multisource BFS (and that doesn't require the graph to be a tree).
But since it is given that the graph is a tree, we can use Tree DP to solve it
*/
void dfs(int u,int p,vector<vector<int>>& g,vector<bool>& marked,vector<int>& dp){
    if(marked[u]){
        dp[u]=0;
    }
    for(auto v:g[u]){
        if(v==p){
            continue;
        }
        dfs(v,u,g,marked,dp);
        dp[u]=(dp[v]==-1?dp[u]:max(dp[u],1+dp[v]));
    }
}

void dfs2(int u,int p,vector<vector<int>>& g,vector<bool>& marked,vector<int>& dp,int m){
    dp[u]=(m==-1?dp[u],max(dp[u],1+m));
    int mx=-1,mxx=-1,first=-1,second=-1;
    for(auto v:g[u]){
        if(v==p){
            continue;
        }
        if(dp[v]>mx){
            second=first;
            mxx=mx;
            first=v;
            mx=dp[v];
        }
        else if(dp[v]>mxx){
            second=v;
            mxx=dp[v];
        }
    }
    for(auto v:g[u]){
        if(v==p){
            continue;
        }
        int x=-1,take=(v==first?mxx:mx);
        x=(take==-1?-1:1+take);
        x=(marked[u]?max(0,x):x);
        x=(m==-1?x:max(1+m,x));
        dfs2(v,u,g,marked,dp,x);
    }
}

void solve(){
    // marked[i] -> True if ith vertex is marked. Else false
    vector<int>dp(n+1,-1);
    // dp[i] -> Stores maximum distance of marked vertices from vertex i finally (-1 if no marked vertices are present in the tree). It does so in 2 stages
    dfs(1,0,g,marked,dp);
    // After doing dfs(), dp[i] stores maximum distance of marked vertices from node i in i's subtree (-1 if no marked vertices are present in i's subtree)
    dfs2(1,0,g,marked,dp,-1);
    // After performing dfs2(), dp[i] stores the required answers (i.e. maximum distance of marked vertices from node i/-1 if no marked vertices present in the tree)
    // Here, the last parameter (passed as -1 initially) is the contribution from the parent of passed node
    for(int i=1;i<=n;i++){
        cout<<dp[i]<<" ";
    }
}
```
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
            ans[v]=ans[u]+(a[u]^a[v])*(su-2*sv);
            dfs2(v,u,g,ans,sz,a);
            // Restoring original sz values (as per old root)
            sz[u]=su;
            sz[v]=sv;
        }
    }
}

/*
The function dfs2() can also be written in a much simplified way as follows (dp vector also needs to be passed)

void dfs2(int u,int p,vector<vector<int>>& g,vector<int>& dp,vector<int>& ans,vector<int>& sz,vector<int>& a){
    for(auto v:g[u]){
        if(v==p){
            continue;
        }
        int old_dpu=dp[u],old_dpv=dp[v],old_szu=sz[u],old_szv=sz[v];
        dp[u]-=((a[u]^a[v])*sz[v]+dp[v]);
        sz[v]=sz[u];
        sz[u]-=old_szv;
        dp[v]+=(a[u]^a[v])*sz[u]+dp[u];
        // We have changed dp and sz values as per the new root v
        ans[v]=dp[v];
        dfs2(v,u,g,dp,ans,sz,a);
        // Restoring old dp and sz values as per old root u
        dp[u]=old_dpu;
        dp[v]=old_dpv;
        sz[u]=old_szu;
        sz[v]=old_szv;
    }
}
*/

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
- Another interesting rerooting problem: You are given a tree consisting of $n$ vertices. Each vertex $v$ is either colored black ($color_{v} = 0$) or colored white ($color_{v} = 1$). For each vertex $v$, calculate the maximum value of $count_{white} - count_{black}$ for a subtree including vertex $v$
```cpp
void dfs(int u,int p,vector<vector<int>>& g,vector<int>& color,vector<int>& dp){
    dp[u]=2*color[u]-1;
    for(auto v:g[u]){
        if(v==p){
            continue;
        }
        dfs(v,u,g,color,dp);
        dp[u]+=max(0,dp[v]);
    }
}

void dfs2(int u,int p,vector<vector<int>>& g,vector<int>& dp,vector<int>& ans){
    for(auto v:g[u]){
        if(v==p){
            continue;
        }
        int old_u=dp[u],old_v=dp[v];
        dp[u]-=max(0,dp[v]);
        dp[v]+=max(0,dp[u]);
        // We have changed all the dp values as if the tree is rooted at vertex v
        ans[v]=dp[v];
        dfs2(v,u,g,dp,ans);
        // Restoring dp values as per old root u
        dp[u]=old_u;
        dp[v]=old_v;
    }
}

vector<int>dp(n+1),ans(n+1);
dfs(1,0,g,color,dp); // Calculating answer for node 1
ans[1]=dp[1];
dfs2(1,0,g,dp,ans); // Calculating answer for all other nodes by rerooting the node to adjacent node of previous root
for(int i=1;i<=n;i++){
    cout<<ans[i]<<" ";
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
- Many Probability/Expectation problems require DP for solving. For example, consider this problem, where two binary strings $a$ and $b$ of length $n$ are given. In one operation, you choose an index $i$ $(1 \leq i \leq n)$ uniformly at random and flip the value of $a_{i}$. Find the expected number of moves to make both string equal for the first time
```cpp
/*
Suppose dp[r] -> Number of expected moves to make both strings equal if r of the total n characters match
Then we have dp[r] = 1 + (r/n)*dp[r-1] + (1-r/n)*dp[r+1]
dp[0] = 1 + dp[1]
dp[n] = 0
Let's define k[i] = dp[0] - dp[i]
Then, we have k[0] = dp[0] - dp[0] = 0, k[1] = dp[0] - dp[1] = 1 and k[n] = dp[0] - dp[n] = dp[0] - 0 = dp[0]
Also, k[r] = dp[0] - dp[r] = k[n] - dp[r] and thus, we have dp[r] = k[n] - k[r]
Thus from the recurrence relation, we have k[n] - k[r] = 1 + (r/n)*(k[n] - k[r-1]) + (1-r/n)*(k[n] - k[r+1]) = 1 + k[n] - (1-r/n)*k[r+1] - (r/n)*k[r-1]
From above, we get 1 + k[r] - (1-r/n)*k[r+1] - (r/n)*k[r-1] = 0 and thus k[r+1] = (n/(n-r))*(1 + k[r] - (r/n)*k[r-1]) = (n + n*k[r] - r*k[r-1])/(n-r)
Thus, our final recurrence relation is k[r+1] = (n + n*k[r] - r*k[r-1])/(n-r) with k[0] = 0 and k[1] = 1
Let m characters match initially. Then answer = dp[m] = k[n] - k[m]
*/
int m=0;
for(int i=0;i<n;i++){
    if(a[i]==b[i]){
        m++;
    }
}
vector<int>k(n+1,0);
k[1]=1;
for(int i=1;i<n;i++){
    k[i+1]=(n+n*k[i]-i*k[i-1])%MOD;
    k[i+1]=(k[i+1]*xp(n-i,MOD-2,MOD))%MOD;
}
int ans=(k[n]-k[m]+MOD)%MOD;
cout<<ans<<'\n';
```
- DP is also used extensively for counting (combinatorial) problems. Consider this problem: There is a row of $n$ tiles and an integer $k$ $(1 \leq k \leq n \leq 5000)$. The tiles are indexed from left to right and the $i^{th}$ tile has color $c_{i}$ $(1 \leq c_{i} \leq n)$. You have to choose some tiles, forming a path $p$. A path $p$ of length $m$ is called nice if the following conditions hold. Calculate number of nice paths of maximum length
    - $k | m$
    - $c_{p_{1}} = c_{p_{2}} = ... = c_{p_{k}}$
    - $c_{p_{k+1}} = c_{p_{k+2}} = ... = c_{p_{2k}}$
    - ...
    - $c_{p_{m-k+1}} = c_{p_{m-k+2}} = ... = c_{p_{m}}$
```cpp
// Assuming 1-based indexing of array c
vector<int>dp1(n+1,0),dp2(n+1,0);
vector<vector<int>>color(n+1,vector<int>(n+1,0));
// dp1[i] -> Length of longest path ending at index i satisfying the given constraints (If path length is l, then dp1[i] stores the value l/k)
// dp2[i] -> Number of paths of maximum length (dp1[i]) ending at index i
// color[i][j] -> Count of elements of color i till index j
// Note that we have used C[i][j] to represent the number of ways of choosing j elements from a pool of i elements
for(int i=1;i<=n;i++){
    for(int j=1;j<=n;j++){
        color[j][i]=color[j][i-1]+(c[i]==j?1:0);
    }
}
int ans=1,mx=0;
for(int i=1;i<=n;i++){
    if(color[c[i]][i]>=k){
        dp1[i]=1;
        dp2[i]=C[color[c[i]][i]-1][k-1];
    }
    for(int j=1;j<i;j++){
        int cnt=color[c[i]][i]-color[c[i]][j];
        if(cnt<k){
            continue;
        }
        if(dp1[j] && dp1[i]<1+dp1[j]){
            dp1[i]=1+dp1[j];
            dp2[i]=dp2[j]*C[cnt-1][k-1];
        }
        else if(dp1[j] && dp1[i]==1+dp1[j]){
            dp2[i]+=dp2[j]*C[cnt-1][k-1];
        }
    }
    if(dp1[i]>mx){
        mx=dp1[i];
        ans=dp2[i];
    }
    else if(dp1[i]==mx && mx){
        ans+=dp2[i];
    }
}
cout<<(mx?ans:1)<<'\n';
```
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

- **Pigeonhole Principle**: A problem illustrating its use: You are given an array $a$ ($0 \leq a_i \leq 10^9$) of length $n$ ($1 \leq n \leq 2\times10^5$). You need to calculate the value of $\prod_{i \neq j}|a_i - a_j|$ modulo $m$ ($1 \leq m \leq 10^3$)
```cpp
/*
Note that if n > m, then there exist distinct indices i and j such that ai mod m = aj mod m
by Pigeonhole principle. In this case, the value of |ai - aj| mod m becomes 0 and the product
also becomes 0. Else, we can bruteforce the value of product since m <= 1000
*/

if(n>m){
    cout<<"0\n";
    return;
}
int ans=1;
for(int i=0;i<n;i++){
    for(int j=i+1;j<n;j++){
        ans=(ans*abs(a[i]-a[j]))%m;
    }
}
cout<<ans<<'\n';
```
- **Stars and Bars method (To determine number of non-negative integer solutions)**: Let the equation be $\Sigma_{i=1}^{r}x_i = n$, where each $x_i \geq 0$. We need to find number of distinct solutions to the given equation. This problem can be modelled as follows. Suppose $n$ identical stars are kept in a straight line. Now, we need to place $(r - 1)$ identical bars to create $r$ partitions. The number of stars to the left of leftmost bar $=$ value of $x_1$. Number of stars to the right of rightmost bar $=$ value of $x_r$. Number of stars between $(i-1)^{th}$ and $i^{th}$ bar (assuming $1$-indexing) $=$ value of $x_i$. Thus the given problem now reduces to finding number of ways to arrange $n$ identical stars and $(r - 1)$ identical bars, and thus equal to $\binom{n+r-1}{n}$
- **Counting number of permutations of a particular string**: Suppose our string contains the alphabets $x_1, x_2, ..., x_n$. Suppose the $i^{th}$ alphabet $x_i$ appears $r_i$ times in the string. Now, number of distinct permutations of the given string is equal to $\binom{r_1+r_2+...+r_n}{r_1} \times \binom{r_2+r_3+...+r_n}{r_2} \times ... \times \binom{r_n}{r_n} = \frac{(r_1+r_2+...+r_n)!}{r_1!r_2!...r_n!}$
- Suppose there are $r_{1}$ objects of type $1$, $r_{2}$ objects of type $2$, $...$, $r_{k}$ objects of type $k$. Suppose objects of type $i$ can be arranged among themselves in $\alpha_{i}$ ways and two objects of different types can be arranged in any order. The total number of ways to arrange these $r_{1} + r_{2} + ... + r_{k}$ objects $=$ $\frac{(r_{1} + r_{2} + ... + r_{k})!}{r_{1}!r_{2}!...r_{k}!}\alpha_{1}\alpha_{2}...\alpha_{k}$. Consider this problem illustrating the use of this technique: There are $n$ people, numbered $1$ to $n$. You are given a graph where an edge between node $u$ and node $v$ means that $u$ and $v$ are friends. Also, the friendship is transitive, that is, if $u$ and $v$ are friends and $v$ and $w$ are friends, then $u$ and $w$ are also friends. You are also given an array $a$ of length $n$ consisting of positive integers. Calculate the number of ways to arrange these $n$ people such that if $u$ and $v$ are friends, then $u$ comes before $v$ iff $a_{u} \leq a_{v}$
```cpp
/*
Note that the graph can be divided into several connected components, where the nodes in each
component are friends of each other. Let there be k such components of sizes g1, g2, ... gk.
Number of ways to arrange these n people = n!/(g1!g2!...gk!) * x(1)x(2)...x(k) where x(i)
represents the number of ways friends in component i can arrange themselves. It is easy to see
that x(i) = product of factorial of frequencies of the array values
*/

void dfs(int u,vector<vector<int>>& graph,vector<bool>& visited,map<int,int>& frequency,vector<int>& a,stack<int>& stk){
    visited[u]=1;
    frequency[a[u]]++;
    for(auto v:g[u]){
        if(!vis[v]){
            dfs(v,graph,visited,frequency,a,stk);
        }
    }
    stk.push(u);
}

void solve(){
    map<int,int>frequency;
    int ans=factorial[n];
    vector<bool>visited(n+1,false);
    stack<int>stk;
    for(int i=1;i<=n;i++){
        if(!vis[i]){
            dfs(i,graph,visited,frequency,a,stk);
            int sz=stk.size();
            ans/=factorial[sz];
            while(!stk.empty()){
                ans*=factorial[frequency[a[stk.top()]]];
                frequency[a[stk.top()]]=0;
                stk.pop();
            }
        }
    }
    cout<<ans<<'\n';
}
```
- **Coefficient of $x^r$ in $(1-x)^{-n} =$** Coefficient of $x^r$ in $((1-x)^{-1})^{n} =$ Coefficient of $x^r$ in $(1+x+x^{2}+...)^{n} =$ Number of integer solutions of $y_{1}+y_{2}+...+y_{n} = r$ where each $y_{i} \geq 0$ is equal to $\binom{n+r-1}{r}$
- **Stirling Numbers**
    - **Unsigned Stirling number of the first kind**: Denoted by $S_{1}(n,k)$, it is defined as number of permutations of $n$ elements with $k$ cycles. Suppose we have $(n+1)$ elements and we have to partition it into $k$ cycles. Now, the $(n+1)^{th}$ element can be introduced at any of the $n$ places or it can form a separate cycle. Thus recurrence relation for $S_{1}(n+1,k)$ is $S_{1}(n+1,k) = nS_{1}(n,k) + S_{1}(n,k-1)$ for $k > 0$. The base case would be $S_{1}(0,0) = 1$ and $S_{1}(n,0) = S_{1}(0,n) = 0$ for $n > 0$
    - **Stirling number of the second kind**: Denoted by $S(n,k)$, it counts the number of ways to partition a set of $n$ labeled objects into $k$ non-empty unlabeled subsets. Suppose we have $(n+1)$ labeled objects and we have to partition it into $k$ non-empty subsets. Now, the $(n+1)^{th}$ object can be introduced into one of the $k$ existing subsets or it can form a separate subset. Thus recurrence relation for $S(n+1,k)$ is $S(n+1,k) = kS(n,k) + S(n,k-1)$ for $0 < k < n$. The base case would be $S(n,n) = 1$ for $n \geq 0$ and $S(n,0) = S(0,n) = 0$ for $n > 0$
- **Burnside's Lemma (Polya's Enumeration Principle)**
- **Catalan Numbers** ($C_{n} = \frac{\binom{2n}{n}}{n+1}$)
- **Inclusion-Exclusion Principle**: The principle of inclusion-exclusion states that for finie sets $A_{1}, A_{2},...,A_{n}$, one has the identity
  $$\bigg |\bigcup_{i=1}^{n} A_{i}\bigg | = \sum_{i=1}^{n} |A_{i}| - \sum_{1 \leq i < j \leq n} |A_{i} \cap A_{j}| + \sum_{1 \leq i < j < k \leq n} |A_{i} \cap A_{j} \cap A_{k}| - ... + (-1)^{n+1}|A_{1} \cap A_{2} ... \cap A_{n}|$$
- We can use Inclusion-Exclusion Principle to calculate the number of derangements of $n$ objects. We need to calculate number of permutations of $n$ objects such that no object appears at its original place. Let $S_{k}$ represent number of permutations of $n$ objects such that the $k^{th}$ object appears at its original place. So number of derangements would be then equal to $n! - |S_{1} \cup S_{2} \cup ... \cup S_{n}|$. Now, by principle of inclusion-exclusion, we have
$$|S_{1} \cup ... \cup S_{n}| = \sum_{i=1}^{n} |S_{i}| - \sum_{1 \leq i < j \leq n} |S_{i} \cap S_{j}| + \sum_{1 \leq i < j < k \leq n} |S_{i} \cap S_{j} \cap S_{k}| - ... + (-1)^{n+1}|S_{1} \cap S_{2} ... \cap S_{n}|$$
$$= \binom{n}{1}(n-1)! - \binom{n}{2}(n-2)! + ... + (-1)^{n+1}\binom{n}{n}0!$$
$$= n![\frac{1}{1!} - \frac{1}{2!} + ... + (-1)^{n+1}\frac{1}{n!}]$$
And thus, the number of derangements is given by
$$n! - \bigg |\bigcup_{i=1}^{n} S_{i}\bigg | = n! - n![\frac{1}{1!} - \frac{1}{2!} + ... + (-1)^{n+1}\frac{1}{n!}] = n!\sum_{i=0}^{n}\frac{(-1)^{i}}{i!}$$
- **Binomial Theorem**: It states that $(a+b)^n = \Sigma_{r=0}^{n}\binom{n}{r}a^{n-r}b^{r}$ where $a, b \in \mathbb{R}$ and $n \in \mathbb{N}$
- **Generating Functions**
- A problem illustrating the use of exponential generating functions:\
**Problem**: How many $n$ digit even numbers ($n > 1$) can be formed using the digits $[1, 2, ..., 9]$ such that digit $9$ appears atleast once and digit $8$ appears an even number of times\
\
**Solution**: We shall consider two separate cases:\
Case $1$: The number ends with $2, 4$ or $6$\
In this case, required count $= 3 \times (n-1)! \times \textrm{Coefficient of }x^{n-1} \textrm{ in expansion of } (1 + x + \frac{x^2}{2!} + ...)^7(1 + \frac{x^2}{2!} + \frac{x^4}{4!} + ...)(x + \frac{x^2}{2!} + \frac{x^3}{3!} + ...)$\
$= 3 \times (n-1)! \times \textrm{Coefficient of }x^{n-1} \textrm{ in expansion of } e^{7x}(\frac{e^{x} + e^{-x}}{2})(e^{x} - 1)$\
$= 3 \times (n-1)! \times \textrm{Coefficient of }x^{n-1} \textrm{ in expansion of } \frac{e^{9x} - e^{8x} + e^{7x} - e^{6x}}{2}$\
$= \frac{3}{2}(9^{n-1} - 8^{n-1} + 7^{n-1} - 6^{n-1})$\
Case $2$: The number ends with $8$\
In this case, required count $= (n-1)! \times \textrm{Coefficient of }x^{n-1} \textrm{ in expansion of } (1 + x + \frac{x^2}{2!} + ...)^7(x + \frac{x^3}{3!} + \frac{x^5}{5!} + ...)(x + \frac{x^2}{2!} + \frac{x^3}{3!} + ...)$\
$= (n-1)! \times \textrm{Coefficient of }x^{n-1} \textrm{ in expansion of } e^{7x}(\frac{e^{x} - e^{-x}}{2})(e^{x} - 1)$\
$= (n-1)! \times \textrm{Coefficient of }x^{n-1} \textrm{ in expansion of } \frac{e^{9x} - e^{8x} - e^{7x} + e^{6x}}{2}$\
$= \frac{1}{2}(9^{n-1} - 8^{n-1} - 7^{n-1} + 6^{n-1})$\
Summing the results of above two cases, our required answer $= 2(9^{n-1} - 8^{n-1}) + 7^{n-1} - 6^{n-1}$
- **Partitions**
- A nice problem and approach to solve it\
**Problem**: Given non-negative integers $n$ and $m$, find number of non-decreasing sequences of length $n$ such that each element in the sequence lies between $0$ and $m$ inclusive\
\
**Solution**: First, we choose $k$ distinct numbers out of $(m + 1)$ numbers. Then we need to arrange these $k$ numbers into $n$ boxes, so number of ways of arranging is equal to number of integer solutions of $x_1 + x_2 + ... + x_k = n$, where each $x_i \geq 1$. So required answer $= \Sigma_{k=1}^{n}\binom{m+1}{k}\binom{n-1}{k-1} = \Sigma_{k=1}^{n}\binom{m+1}{k}\binom{n-1}{n-k} =$ Coefficient of $x^n$ in $(1+x)^{m+1} \cdot (1+x)^{n-1} =$ Coefficient of $x^n$ in $(1+x)^{m+n} = \binom{m+n}{n}$
- Another nice problem\
**Problem**: Find number of integer solutions of $x_{1} + x_{2} + ... + x_{k} \leq n$, where each $x_{i} \geq y_{i} \geq 0$ and $y_{1} + y_{2} + ... + y_{k} \leq n$\
\
**Solution**: Define $z_{i} = x_{i} - y_{i}$. The given inequality reduces to $z_{1} + z_{2} + ... +z_{k} \leq n - y_{1} - y_{2} ... - y_{k}$, where each $z_{i} \geq 0$. Let's define another dummy variable $z_{k+1}$ to consume the slack in the inequality. Thus, we need to determine the number of integer solutions of $z_{1} + z_{2} + ... + z_{k+1} = n - y_{1} - y_{2} ... - y_{k}$, with each $z_{i} \geq 0$, which is equal to $\binom{n + k - y_{1} - y_{2} ... - y_{k}}{k}$
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
- Another nice problem\
**Problem**: The score of a binary string of length $n$ is defined as the count of blocks formed by consecutive $0$'s and $1$'s. For example the score of $1001110100$ is $6$ as there are $6$ blocks, $1$, $00$, $111$, $0$, $1$ and $00$. Find the total sum of scores of all possible binary strings of length $n$\
\
**Solution 1**: Let us find the count of binary strings of length $n$ which have score $k$. Let $x_{i}$ denote the number of characters in the $i^{th}$ block of the string. Thus, number of strings of length $n$ having score $k$ $=$ $2$ $\times$ Number of integer solutions of $x_{1} + x_{2} + ... + x_{k} = n$, where each $x_{i} > 0$. We have multiplied by $2$ because the string can either start with a $0$ or a $1$. Thus, our answer is given by
$$W(n) = \sum_{k=1}^{n}2k\binom{n-1}{k-1} = \sum_{k=1}^{n}2k\frac{(n-1)!}{(k-1)!(n-k)!} = \sum_{k=1}^{n}\frac{2k^{2}}{n}\frac{n(n-1)!}{k(k-1)!(n-k)!} = \sum_{k=1}^{n}\frac{2k^{2}}{n}\frac{n!}{k!(n-k)!}$$
which is equivalent to
$$W(n) = \sum_{k=1}^{n}\frac{2k^{2}}{n}\binom{n}{k} = \frac{2}{n}\sum_{k=1}^{n}k^{2}\binom{n}{k}$$
To find the closed form of the summation, consider
$$f(x) = (1+x)^{n} = \sum_{k=0}^{n}\binom{n}{k}x^{k}$$
Differentiating both sides with respect to $x$, we get
$$f'(x) = n(1+x)^{n-1} = \sum_{k=1}^{n}k\binom{n}{k}x^{k-1}$$
Multiplying on both sides by $x$, we get
$$xf'(x) = nx(1+x)^{n-1} = \sum_{k=1}^{n}k\binom{n}{k}x^{k}$$
Differentiating both sides again wrt $x$, we get
$$f'(x) + xf''(x) = n[(1+x)^{n-1} + (n-1)x(1+x)^{n-2}] = \sum_{k=1}^{n}k^{2}\binom{n}{k}x^{k-1}$$
Plugging $x = 1$, we get
$$f'(1) + f''(1) = n[2^{n-1} + (n-1)2^{n-2}] = \sum_{k=1}^{n}k^{2}\binom{n}{k}$$
Thus, we have
$$\sum_{k=1}^{n}k^{2}\binom{n}{k} = n[2^{n-1} + (n-1)2^{n-2}] = n(n+1)2^{n-2}$$
Thus, our required answer is given by
$$W(n) = \frac{2}{n}\sum_{k=1}^{n}k^{2}\binom{n}{k} = \frac{2}{n}n(n+1)2^{n-2} = (n+1)2^{n-1}$$
\
**Solution 2**: Let $W(n)$ be the required answer. Consider the action of appending a $0$ or a $1$ to all binary strings of length $(n-1)$. If we append the same digit as the last digit of the $(n-1)$ length string, we get $W(n-1)$ added to the score. If we append the other digit, then the count of blocks increases by $1$ for all $2^{n-1}$ strings and thus it increases our score by $W(n-1) + 2^{n-1}$. Thus, we get the recurrence relation
$$W(n) = 2W(n-1) + 2^{n-1}, n > 1$$
Also $W(1) = 2$. Solving the above recurrence gives the same answer $W(n) = (n+1)2^{n-1}$
- Another problem\
**Problem**: For an array $b$ of length $m$, define $f(b) = \Sigma_{i=1}^{m-1}(-1)^{i+1}|b_{i} - b_{i+1}|^{7}$. You are given an array $a$ of length $n$. Find sum of $f(s)$ over over all non-empty subsequences $s$ of $a$\
\
**Solution**: Let elements $a_i$ and $a_j$ form consecutive elements in a subsequence. Suppose $i > 1$. Now, contribution of this pair to the sum $= |a_i - a_j|^7.2^{n-j}$ if an even number of elements occur in the subsequence before index $i$ else the contrbution is $-|a_i - a_j|^7.2^{n-j}$. So, contribution of this pair $= (\binom{i-1}{0} + \binom{i-1}{2} + ...)|a_i - a_j|^7.2^{n-j} - (\binom{i-1}{1} + \binom{i-1}{3} + ...)|a_i - a_j|^7.2^{n-j} = 0$ as $\binom{k}{0} - \binom{k}{1} + \binom{k}{2} - ... + (-1)^k\binom{k}{k} = (1 - 1)^k = 0$. Thus for all pairs $(i, j)$ such that $i > 1$, the contribution of this pair to the sum is $0$. Hence required answer $= \Sigma_{j=2}^{n}|a_1 - a_j|^7.2^{n-j}$
- Another interesting problem\
**Problem**: Show that $\Sigma_{k=1}^{\lfloor\frac{n+1}{2}\rfloor}k\binom{n+1}{2k} = (n+1)2^{n-2}$ for all $n > 0$\
\
**Solution 1**: The case $n = 1$ is easy to verify. For $n > 1$, it is easy to see from one of the previous problems that $RHS$ is actually equal to count of blocks formed by $0's$ across all binary strings of length $n$. The other way to calculate this is to find count of binary strings with exactly $k$ blocks of $0's$ in it and then sum the expression from $k = 0$ to $k = \lfloor\frac{n+1}{2}\rfloor$, as there can't be more than $\lfloor\frac{n+1}{2}\rfloor$ blocks of $0's$. Now, to calculate the count of binary strings with exactly $k$ blocks of $0's$, the count is equal to the number of integer solutions of $y_{0} + x_{1} + y_{1} + x_{2} + ... + x_{k} + y_{k} = n$ where $y_{0}, y_{k} \geq 0$ and rest of the variables are $> 0$. This count is exactly equal to $\binom{n+1}{2k}$. So total number of blocks of $0's$ across all binary strings of length $n = \Sigma_{k=0}^{\lfloor\frac{n+1}{2}\rfloor}k\binom{n+1}{2k} = \Sigma_{k=1}^{\lfloor\frac{n+1}{2}\rfloor}k\binom{n+1}{2k} = LHS$\
\
**Solution 2**: $$LHS = \sum_{k=1}^{\lfloor\frac{n+1}{2}\rfloor}k\binom{n+1}{2k} = \sum_{k=1}^{\lfloor\frac{m}{2}\rfloor}k\binom{m}{2k} \textrm{where } m = n + 1$$
We have
$$f(x) = (1+x)^m + (1-x)^m = \sum_{k=0}^{\lfloor\frac{m}{2}\rfloor}2\binom{m}{2k}x^{2k}$$
Differentiating with respect to $x$, we get
$$f'(x) = m[(1+x)^{m-1} - (1-x)^{m-1}] = \sum_{k=1}^{\lfloor\frac{m}{2}\rfloor}4k\binom{m}{2k}x^{2k-1}$$
Dividing by $4$ and putting $x = 1$, we get
$$\frac{f'(1)}{4} = \frac{m}{4}.2^{m-1} = m.2^{m-3} = \sum_{k=1}^{\lfloor\frac{m}{2}\rfloor}k\binom{m}{2k}$$
and thus
$$LHS = \sum_{k=1}^{\lfloor\frac{m}{2}\rfloor}k\binom{m}{2k} = m.2^{m-3} = (n+1)2^{n-2} = RHS$$
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
- A problem illustrating the use of above technique: You are given an array $a$ consisting of $n$ distinct integers $a_{1}, a_{2}, ..., a_{n}$. Define the beauty of an array $p_{1}, p_{2}, ..., p_{k}$ as the minimum amount of time needed to sort this array using an arbitrary number of range-sort operations. In each range-sort operation, you choose two integers $l$ and $r$ $(1 \leq l < r \leq k)$. Then sort the subarray $p_{l}, p_{l+1}, ..., p_{r}$ in $(r-l)$ seconds. Calculate sum of beauty over all subarrays of $a$
```cpp
/*
Since time to sort a subarray is 1 less than its length, therefore to sort the array in
minimum time, we need to sort as many non overlapping subarrays as possible so that resulting
array is sorted. For example to sort [3, 1, 2, 5, 4], we would sort the subarray [3, 1, 2]
and [5, 4] independently. The resulting array would be sorted. Thus minimum time to sort a
subarray a[l, r] = (r - l) - count of positions k (l <= k < r) such that maximum element in
subarray a[l, k] < minimum element in subarray a[k + 1, r]. To find total time across all
subarrays, we can first sum up the value (r - l) for all subarrays a[l, r]. Now, from this,
we need to subtract the value as mentioned in previous equation. We can do this by finding
the number of subarrays which satisfy previous inequality for each position in the array. The
sum of contributions of each position in the array can be subtracted
*/
vector<int>ple(n,-1),nle(n,n),pge(n,-1);
// ple[i] -> Previous lesser element for index i
// nle[i] -> Next lesser element for index i
// pge[i] -> Previous greater element to the left of ple[i] for index i
stack<int>stk;
for(int i=0;i<n;i++){
    while(!stk.empty() && a[stk.top()]>a[i]){
        nle[stk.top()]=i;
        stk.pop();
    }
    stk.push(i);
}
while(!stk.empty()){
    stk.pop();
}
for(int i=n-1;i>=0;i--){
    while(!stk.empty() && a[stk.top()]>a[i]){
        ple[stk.top()]=i;
        stk.pop();
    }
    stk.push(i);
}
while(!stk.empty()){
    stk.pop();
}
stack<int>stk2;
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
int total=0;
// total = 1*(n-1) + 2*(n-2) + ... + (n-1)*1
for(int i=1;i<n;i++){
    total+=(i*(n-i));
}
/*
From total, we will subtract contribution of a[i]. Contribution of a[i] = Number of triplets (l, k, r) such that
max(a[l]...a[k]) < min(a[k+1]...a[r]) = a[i]. Note that this approach works since all integers in array a are distinct
*/
for(int i=1;i<n;i++){
    // Subtract contribution of a[i]
    int x=pge[i];
    int y=ple[i];
    int z=nle[i];
    // All subarrays with left end in [x+1, y] and right end in [i, z-1] are part of a[i]'s contribution
    total-=((z-i)*(y-x));
}
cout<<total<<'\n';
```
- Sometimes, to calculate count of objects which satisfy property $X$, we instead calculate count of objects which don't satisfy property $X$ (if it's easier) and then subtract it from total objects. A problem illustrating this and the use of contribution technique: For a sequence $X$, let $f(X) =$ minimum number of elements to be modified to make $X$ a palindrome. Given an array $a$ of length $n$ $(1 \leq a_{i} \leq m)$, calculate sum of $f(X)$ over all subarrays of $a$
```cpp
/*
We shall assume 1-based indexing for array a.
We will count contribution of each pair (l,r) (l<r). A pair (l,r) contributes 1 for a subarray
(L,R) if l-L==R-r and a[l]!=a[r]. Let's call a pair (l,r) good if a[l]==a[r]. We need to calculate
total count of bad pairs across all subarrays. Now, total count of bad pairs across all subarrays
= total count of pairs (l,r) across all subarrays - total count of good pairs across all subarrays.
To evaluate second part of RHS, instead of evaluating number of good pairs for each subarray, we will
instead evaluate how many subarrays each good pair contributes to. We note that a good pair (l,r)
contributes to min(l,n-r+1) subarrays
*/
vector<vector<int>>p(m+1);
for(int i=1;i<=n;i++){
    p[a[i]].push_back(i);
}
int ans=0;
// We first calculate total count of pairs across all subarrays
for(int i=1;i<=n;i++){
    ans+=(n-i+1)*(i/2); // Contribution of subarray of length i (There are (n-i+1) subarrays of length i and each such subarray has floor(i/2) pairs)
}
for(int i=1;i<=m;i++){
    // We will subtract count of good pairs (x,y) such that a[x] == a[y] == i from ans
    int l=0,r=(int)p[i].size()-1;
    // We will use two pointer method to calculate sum of min(p[i][l],n-p[i][r]+1) for all pairs in array p[i]
    while(l<r){
        if(p[i][l]<=n-p[i][r]+1){ // Then it is also true for pairs (l,l+1), (l,l+2) ... (l,r-1)
            ans-=(r-l)*p[i][l];
            l++;
        }
        else{ // Also true for pairs (l+1,r), (l+2,r) .. (r-1,r)
            ans-=(r-l)*(n-p[i][r]+1);
            r--;
        }
    }
}
cout<<ans<<'\n';
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
- This algorithm works because if we write $a = bq + r$ such that $0 \leq r < b$, then we have $gcd(a, b) = gcd(b, r)$. Let $gcd(a, b) = g$ and $gcd(b, r) = g'$. Since $gcd(a, b) = g$, therefore $g | (a - bq)$ and hence $g | r$. Note that $g | b$ and $g | r$ and since $gcd(b, r) = g'$, therefore we have $g \leq g'$. Also note that $gcd(b, r) = g'$ implies $g' | (bq + r)$ and hence $g' | a$. Since $g' | a$ and $g' | b$ and $gcd(a, b) = g$, therefore $g' \leq g$. We have $g \leq g'$ and $g' \leq g$ and thus $g = g'$ and hence $gcd(a, b) = gcd(b, r)$
- $LCM(a,b)$ can be determined as $\frac{(a \times b)}{gcd(a,b)}$
- Time complexity is $O(log(min(a, b)))$
- Suppose you have to calculate $GCD$ of an array of integers of length $n$ where maximum element is $m$. Then the time complexity of it would be $O(n + log$ $m)$

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
- Time complexity is $O(log$ $n)$

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
- From this, it is easy to observe that number of factors of $N$ is bounded above by $2\sqrt{N}$

**7.6. Modular inverse**

- Calculating modular inverse when $MOD$ is a prime is straightforward. We can use Fermat's Little Theorem. Suppose we have to find inverse of $a$ modulo $MOD$ where $MOD$ is prime. Then $a^{-1} = a^{MOD-2}$ $modulo$ $MOD$. If $MOD$ is not prime, then we have to use Extended Euclidean Algorithm to find the multiplicative inverse. Note that the inverse of $a$ modulo $MOD$ exists iff $gcd(a, MOD) = 1$
- Extended Euclidean Algorithm
```cpp
int extended_euclidean(int a,int b,int* x,int* y){
    int x2=1,y2=0,x1=0,y1=1,r2=a,r1=b;
    while(r1){
        int q=r2/r1
        int r=r2%r1;
        int nx=x2-q*x1;
        int ny=y2-q*y1;
        r2=r1;
        r1=r;
        x2=x1;
        x1=nx;
        y2=y1;
        y1=ny;
    }
    *x=x2;
    *y=y2;
    return r2;
}
```
- To find modular inverse of $a$ when $MOD$ is not prime (but $gcd(a, MOD) = 1)$, we can use Extended Euclidean Algorithm to find $x$ and $y$ such that $ax + MODy = gcd(a, MOD) = 1$. Now, taking remainder with $MOD$ on both sides, we get $(ax)$ $modulo$ $MOD$ = $1$ and thus $x$ is the modular inverse of $a$ $modulo$ $MOD$

**7.7. Some general notes and ideas**

- Suppose we are given an array of $n$ integers $[a_1, a_2, ..., a_n]$ and a prime number $p$. Suppose we need to check whether for every integer $x > 0$, $p$ divides atleast $2$ of $[a_1+x, a_2+x, ..., a_n+x]$. To check this, we reduce our array to $[a_1$ $mod$ $p, a_2$ $mod$ $p, ..., a_n$ $mod$ $p]$. Now, if $min(cnt_0, cnt_1, ..., cnt_{p-1}) > 1$ in our reduced array, then $p$ divides atleast $2$ of $[a_1+x, a_2+x, ..., a_n+x]$ $\forall x > 0$. Else there exists some $x$ such that $p$ divides atmost $1$ of $[a_1+x, a_2+x, ..., a_n+x]$
- The Harmonic series $\Sigma_{i=1}^{n} \frac{1}{i} = O(log$ $n)$ is very useful in determining time complexities of many algorithms
- There are $O(\sqrt{n})$ distinct values of $\lfloor\frac{n}{i}\rfloor$ when $i$ varies from $1$ to $n$
- The maximum value of $j$ such that $\lfloor\frac{n}{i}\rfloor = \lfloor\frac{n}{j}\rfloor =$ $x$ (say) is given by $j = \lfloor\frac{n}{x}\rfloor$
- $gcd(kp, kq) = k \cdot gcd(p, q)$ and $lcm(kp, kq) = k \cdot lcm(p,q)$
- Suppose prime factorization of a number $N = p_1^{\alpha_1}p_2^{\alpha_2}...p_k^{\alpha_k}$
    - Number of factors of $N = \sigma_{0}(N) = (1+\alpha_1)(1+\alpha_2)...(1+\alpha_k)$
    - Sum of factors of $N = \sigma_{1}(N) = (\frac{p_1^{\alpha_1+1}-1}{p_1-1})(\frac{p_2^{\alpha_2+1}-1}{p_2-1})...(\frac{p_k^{\alpha_k+1}-1}{p_k-1})$
    - Product of factors of $N = \sqrt{N^{\textrm{Number of factors of N}}} = \sqrt{N^{\sigma_{0}(N)}} = N^{\frac{\sigma_{0}(N)}{2}}$
    - Euler's totient function $\varphi(N) = N(1-\frac{1}{p_1})(1-\frac{1}{p_2})...(1-\frac{1}{p_k})$
- Bezout's Identity: Let $a$ and $b$ be integers with $gcd$ $d$. Then there exist integers $x$ and $y$ such that $ax + by = d$. Moreover, the integers of the form $az + bt$ are exactly the multiples of $d$ ($d$ is the smallest positive integer in the set $S = [ ax + by : x,y \in \mathbb{Z} \textrm{ and } ax + by > 0 ]$)
- From above, it is easy to observe that a Linear Diophantine Equation $Ax + By = C$ has a solution iff $gcd(A, B) | C$. Let $gcd(A, B) = g$. Then the solutions to original equation are also the solutions to the equation $\frac{A}{g}x + \frac{B}{g}y = \frac{C}{g}$. Since $gcd(\frac{A}{g}, \frac{B}{g}) = 1$, therefore a solution to the equation $\frac{A}{g}x + \frac{B}{g}y = 1$ can be found using Extended Euclidean Algorithm. Let that solution be $(X, Y)$. Thus one solution of equation $\frac{A}{g}x + \frac{B}{g}y = \frac{C}{g}$ is $(\frac{C}{g}X, \frac{C}{g}Y)$ and this is also a solution to the equation $Ax + By = C$. The general solution would be $(\frac{C}{g}X + k\frac{B}{g}, \frac{C}{g}Y - k\frac{A}{g})$, where $k$ is any integer
- Chicken McNugget Theorem: For any two relatively prime positive integers $m$ and $n$, the greatest integer that cannot be written in the form $am + bn$ for non-negative integers $a, b$ is $mn - m - n$
- Suppose $n$ events occur after every $t_{1}, t_{2}, ..., t_{n}$ seconds and suppose, all these events occur together at $t = 0$. These events will occur again after every $lcm(t_{1}, t_{2}, ..., t_{n})$ seconds
- Suppose prime factorization of numbers $N_{1}$ and $N_{2}$ is as follows: $N_{1} = p_1^{\alpha_1}p_2^{\alpha_2}...p_k^{\alpha_k}$ and $N_{2} = p_1^{\beta_1}p_2^{\beta_2}...p_k^{\beta_k}$. Then
    - $gcd(N_{1}, N_{2}) =  p_1^{min(\alpha_1,\beta_1)}p_2^{min(\alpha_2,\beta_2)}...p_k^{min(\alpha_k,\beta_k)}$
    - $lcm(N_{1}, N_{2}) =  p_1^{max(\alpha_1,\beta_1)}p_2^{max(\alpha_2,\beta_2)}...p_k^{max(\alpha_k,\beta_k)}$
    - $gcd(N_{1}, N_{2}) \times lcm(N_{1}, N_{2}) = p_1^{\alpha_1 + \beta_1}p_2^{\alpha_2 + \beta_2}...p_k^{\alpha_k + \beta_k} = N_{1}N_{2}$
- Suppose $p$ is a prime number and $n \in \mathbb{N}$. The highest power of $p$ which divides $n!$ is given by $\Sigma_{i=1}^{\infty}\lfloor\frac{n}{p^i}\rfloor$
- Let $a_{1}, a_{2}, ..., a_{n}$ be $n$ positive integers. Then the greatest integer $d$ such that the remainder left on dividing $a_{i}$ by $d$ is same for all $n$ integers is given by $d = gcd(|a_{2} - a_{1}|, |a_{3} - a_{2}|, ..., |a_{n} - a_{n-1}|)$
- Euler's Theorem: If $gcd(a, n) = 1$, then $a^{\varphi(n)} = 1$ $(mod$ $n)$. Fermat's little theorem is a special case of this where we have a prime $p$ instead of $n$ and thus $\varphi(p) = p - 1$
- The sum of all numbers that are $< n$ and coprime to $n$ is given by $\frac{\varphi(n)}{2}n$. This is because if $m (< n)$ is coprime to $n$, then $n - m$ is also coprime to $n$
- Count of numbers $a$ $(1 \leq a \leq n)$ such that $gcd(a, n) = d$ (where obviously $d | n$ holds) $=$ Count of numbers $a$ $(1 \leq a \leq n)$ such that $gcd(\frac{a}{d}, \frac{n}{d}) = 1$ and is thus $=$ $\varphi(\frac{n}{d})$ (By definition of $\varphi(\frac{n}{d}))$
- From previous observation, it is easy to see that $\varphi(d)$ (where $d | n$) $=$ Count of elements $a$ $(1 \leq a \leq n)$ such that $gcd(a, n) = \frac{n}{d}$. Thus summing $\varphi(d)$ over all possible divisors of $n$, we should get $n$. Thus, we have $\Sigma_{d | n}\varphi(d) = n$
- A function $f(n)$ is called a multiplicative function if
    - $f(n)$ is defined for all positive integers $n$
    - $f(1) = 1$
    - $f(mn) = f(m)f(n)$ whenever $gcd(m, n) = 1$
- Examples of multiplicative functions include $\varphi(n)$, $\sigma_{0}(n)$ and $\sigma_{1}(n)$
- Divisor sum theorem of multiplicative functions: Let $f(n)$ be a multiplicative function and let $F(n) = \Sigma_{d | n}f(d)$. Then $F(n)$ is also multiplicative. This can be proved as follows: Let $m, n$ be positive integers such that $gcd(m, n) = 1$. Then $F(mn) = \Sigma_{d | mn}f(d)$. Let $d = rs$ where $r | m$ and $s | n$. Since $gcd(m, n) = 1$, therefore $gcd(r, s) = 1$. So $F(mn) = \Sigma_{d | mn}f(d) = \Sigma_{r | m, s | n}f(rs) = \Sigma_{r | m, s | n}f(r)f(s) = \Sigma_{r | m}f(r)\Sigma_{s | n}f(s) = F(m)F(n)$. Thus we have proved $F(mn) = F(m)F(n)$ whenever $gcd(m, n) = 1$ and hence $F(n)$ is also a multiplicative function
