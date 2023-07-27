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

- In Dynamic Programming, we build the DP array as per the recurrence relation we have
- DP on trees problems usually require us to consider the tree as a rooted tree and then do DFS while maintaining DP vector(s)
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
- We need not visit all the states of DP
- Many Expectations problems require DP for solving
- DP with bitmasking is often used in problems involving recurrence relations on subsets of a set

**6. Combinatorics**

- Pigeonhole Principle
- **Stars and Bars method (To determine number of non-negative integer solutions)**: Let the equation be $\Sigma_{i=1}^{r}x_i = n$, where each $x_i \geq 0$. We need to find number of distinct solutions to the given equation. This problem can be modelled as follows. Suppose $n$ identical stars are kept in a straight line. Now, we need to place $(r - 1)$ identical bars to create $r$ partitions. The number of stars to the left of leftmost bar = value of $x_1$. Number of stars to the right of rightmost bar = value of $x_r$. Number of stars between $(i-1)^{th}$ and $i^{th}$ bar (assuming 1-indexing) = value of $x_i$. Thus the given problem now reduces to finding number of ways to arrange $n$ identical stars and $(r - 1)$ identical bars, and thus equal to $^{n+r-1}C_n$
- **Counting number of permutations of a particular string**: Suppose our string contains the alphabets $x_1, x_2, ..., x_n$. Suppose the $i^{th}$ alphabet $x_i$ appears $r_i$ times in the string. Now, number of distinct permutations of the given string is equal to $^{r_1+r_2+...+r_n}C_{r_1} \times ^{r_2+r_3+...+r_n}C_{r_2} \times ... \times ^{r_n}C_{r_n} = \frac{(r_1+r_2+...+r_n)!}{r_1!r_2!...r_n!}$
- Coefficient of $x^r$ in $(1-x)^{-n}$ = $^{n+r-1}C_r$
- Stirling Numbers
- Burnside's Lemma (Polya's Enumeration Principle)
- Catalan Numbers
- Inclusion-Exclusion Principle
- **Binomial Theorem**: It states that $(a+b)^n = \Sigma_{r=0}^{n}{^{n}}C_ra^{n-r}b^{r}$ where $a, b \in \mathbb{R}$ and $n \in \mathbb{N}$
- Generating Functions
- Partitions
- A nice problem and approach to solve it\
**Problem**: Given non-negative integers $n$ and $m$, find number of non-decreasing sequences of length $n$ such that each element in the sequence lies between $0$ and $m$ inclusive\
\
**Solution**: First, we choose $k$ distinct numbers out of $(m + 1)$ numbers. Then we need to arrange these $k$ numbers into $n$ boxes, so number of ways of arranging is equal to number of integer solutions of $x_1 + x_2 + ... + x_k = n$, where each $x_i \geq 1$. So required answer $= \Sigma_{k=1}^{n}{^{m+1}}C_k \cdot ^{n-1}C_{k-1} = \Sigma_{k=1}^{n}{^{m+1}}C_k \cdot ^{n-1}C_{n-k} =$ Coefficient of $x^n$ in $(1+x)^{m+1} \cdot (1+x)^{n-1} =$ Coefficient of $x^n$ in $(1+x)^{m+n} = ^{m+n}C_n$
- There is a common combinatorial trick for counting: We change the perspective to count. For example, suppose we have to count the number of good objects of type $A$ each object of type $B$ yields. Another way to count this is as follows: We count how many objects of type $B$ yield each of the possible good objects of type $A$. So basically, the code changes as follows 
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
- Computing $^nC_r$ modulo $MOD$
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
- Many combinatorial problems require DP for solving
- Some useful identities
    - $^nC_r + ^nC_{r-1} = ^{n+1}C_r$ (Suppose there are $n$ normal objects and $1$ special object. Now, $RHS =$ Number of ways of choosing $r$ objects out of these $n+1$ objects $=$ Number of ways of choosing $r$ objects by excluding the special object $+$ Number of ways of choosing $r$ objects by including the special object = $^nC_r + ^nC_{r-1} = LHS$)
    - $\Sigma_{i=r}^{n}{^{i}}C_r = ^{n+1}C_{r+1}$ $(LHS =$ Coefficient of $x^r$ in $\Sigma_{i=r}^{n}(1+x)^i =$ Coefficient of $x^r$ in $\(1+x)^r\frac{(1+x)^{n-r+1}-1}{(1+x)-1}$ = Coefficient of $x^r$ in $\frac{(1+x)^{n+1}-(1+x)^r}{x}$ = Coefficient of $x^{r+1}$ in $(1+x)^{n+1}-(1+x)^r = ^{n+1}C_{r+1} - 0 = RHS$)
    - $\Sigma_{k=0}^{r}{^{m}}C_k\cdot^{n}C_{r-k} = ^{m+n}C_r$ $(LHS =$ Coefficient of $x^r$ in $(1+x)^m\cdot(1+x)^n$ = Coefficient of $x^r$ in $(1+x)^{m+n}$ = $^{m+n}C_r = RHS)$

**7. Number Theory**

- Fermat's Little Theorem
- Euler's Totient Function
- Chinese Remainder Theorem
- Matrix Exponentiation
  
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
- Many problems can be solved by modelling the problem into a known graph problem and then apply known algorithms on this graph like BFS, DFS, Dijkstra, etc
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
- We can find minimum xor of two integers in an array by sorting the array and then finding xor between consecutive elements in the array and taking minimum among all the values obtained. Another method to do the same is by using trie

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
  $s_0s_1...s_{len^{'}-1} = s_{len-len^{'}}s_{len-len{'}+1}...s_{len-1}$ (By equations $\oplus$ and $\odot$)\
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
- $lps$ array can be used to find longest palindromic prefix of a string $s$. Suppose $len(s) = n$. We concatenate $s$ with $rev(s)$. Now, we compute $lps$ array for this new string. The value of $lps[2n-1]$ would be the answer

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
    ll sz;
    vector<ll>sums;
    
    void init(ll n){
        sz=1;
        while(sz<n){
            sz<<=1;
        }
        sums.assign(2*sz,0LL);
    }
    
    void build(vector<ll>& a,ll x,ll lx,ll rx){
        if(rx==lx+1){
            if(lx<(ll)a.size()){
                sums[x]=a[lx];
            }
            return;
        }
        ll m=lx+(rx-lx)/2;
        build(a,2*x+1,lx,m);
        build(a,2*x+2,m,rx);
        sums[x]=sums[2*x+1]+sums[2*x+2];
    }
    
    void set(ll i,ll v,ll x,ll lx,ll rx){
        if(rx==lx+1){
            sums[x]=v;
            return;
        }
        ll m=lx+(rx-lx)/2;
        if(i<m){
            set(i,v,2*x+1,lx,m);
        }
        else{
            set(i,v,2*x+2,m,rx);
        }
        sums[x]=sums[2*x+1]+sums[2*x+2];
    }
    
    ll sum(ll l,ll r,ll x,ll lx,ll rx){
        if(lx>=r || rx<=l){
            return 0;
        }
        if(lx>=l && rx<=r){
            return sums[x];
        }
        ll m=lx+(rx-lx)/2;
        ll s1=sum(l,r,2*x+1,lx,m);
        ll s2=sum(l,r,2*x+2,m,rx);
        return s1+s2;
    }
};
```
- General segment tree template looks something like this
```cpp
struct item{

};

struct segtree{
    ll sz;
    vector<item>values;
    
    item merge(item a,item b){
    	// Merged value of item a and item b
    }
    
    void init(ll n){
        sz=1;
        while(sz<n){
            sz<<=1;
        }
        values.assign(2*sz);
    }
    
    void build(vector<ll>& a,ll x,ll lx,ll rx){
        if(rx==lx+1){
            if(lx<(ll)a.size()){
                // values[x]=;
            }
            return;
        }
        ll m=lx+(rx-lx)/2;
        build(a,2*x+1,lx,m);
        build(a,2*x+2,m,rx);
        values[x]=merge(values[2*x+1],values[2*x+2]);
    }
    
    void set(ll i,ll v,ll x,ll lx,ll rx){
        if(rx==lx+1){
            // values[x]=v;
            return;
        }
        ll m=lx+(rx-lx)/2;
        if(i<m){
            set(i,v,2*x+1,lx,m);
        }
        else{
            set(i,v,2*x+2,m,rx);
        }
        values[x]=merge(values[2*x+1],values[2*x+2]);
    }
    
    item calc(ll l,ll r,ll x,ll lx,ll rx){
        if(lx>=r || rx<=l){
            return NEUTRAL_VALUE;
        }
        if(lx>=l && rx<=r){
            return values[x];
        }
        ll m=lx+(rx-lx)/2;
        item x1=calc(l,r,2*x+1,lx,m);
        item x2=calc(l,r,2*x+2,m,rx);
        return merge(x1,x2);
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
    ll sz;
    vector<ll>heap;
    
    void build_heap(vector<ll>& a){
        ll n=a.size();
        sz=n;
        heap.resize(n);
        for(ll i=0;i<n;i++){
            heap[i]=a[i];
        }
        for(ll i=n/2-1;i>=0;i--){
            min_heapify(i);
        }
    }
    
    void min_heapify(ll i){
        ll l=2*i+1,r=2*i+2;
        ll smallest=i;
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
    
    void decrease_key(ll i,ll val){
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
- Many problems require some careful observation to solve it. For example, observations related to parity, binary/n-ary representation of a number and some invariants/monovariants
- Many problems use the concept used in Huffman Coding (choosing $k$ maximum / $k$ minimum values and combining them). It is implemented using priority queue
- In many problems, a multiset can work as a priority queue. It not only supports finding minimum and maximum elements but also removing a particular element from the multiset. The time complexity of all those operations is $O(log$ $n)$
- In many problems, two instances of same data structure are required to simulate the process mentioned in the problem with better complexity. For example, using two instances of multiset/priority queue to maintain lower and upper half of sorted elements, etc.
- The constraints of the problem provide helpful information. For example, many problems involving Bitmask DP have extremely small constraints to allow programs having exponential time complexity. Problems having very large constraints often involve binary search ($O(log$ $n)$ complexity) or some $O(1)$ computation
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
- Suppose an array of positive integers is given. We need to calculate $k^{th}$ largest sum that can be made from these integers where each integer can be chosen any number of times. If multiple sets have same sum, the sum is counted only once. This is done as follows
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
