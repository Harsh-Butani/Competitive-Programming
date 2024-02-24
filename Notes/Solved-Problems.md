## Problem 1 (Binary Search + DP + Two pointers)

Given an array of numbers $a_1, a_2, ... a_n$ ($1 \leq n \leq 10^5, 1 \leq a_i \leq 10^9$). Your task is to block some elements of the array in order to minimize its cost. Suppose you block the elements with indices $1 \leq b_1 < b_2 ... < b_m \leq n$. The cost of the array is calculated as maximum of
- the sum of blocked elements, i.e, $a_{b_1} + a_{b_2} + ... + a_{b_m}$
- the maximum sum of segments into which the array is divided when blocked elements are removed\
\
Calculate minimum cost of the array after blocking

## Solution to Problem 1
```cpp
/*
We shall binary search on the minimum cost. In order to check whether a cost <= mid is achievable or not,
we will use DP. The details are given in the code 
*/
int lo=0,hi=1e14,mid,ans;
while(lo<=hi){
    mid=lo+(hi-lo)/2;
    vector<int>dp(n+1,1e14);
    /* dp[i] -> Minimum sum of blocked elements till index i (element at index i is blocked) such that
       all segments upto index i have sum <= mid
       dp[i] = a[i] + min(dp[j] where j satisfies a[j+1] + a[j+2] + ... + a[i-1] <= mid)
       We shall store the list of the dp[j] values satisfying the above criteria in a
       multiset using two pointers */
    dp[0]=0;
    multiset<int>mst;
    int j=0;
    mst.insert(dp[0]);
    int left_sum=0,right_sum=0;
    /* left_sum stores the value of a[1] + a[2] + ... + a[j] and right_sum stores the value of
       a[1] + a[2] + ... + a[i-1] */
    for(int i=1;i<=n;i++){
        while(right_sum-left_sum>mid){
            mst.erase(mst.find(dp[j]));
            j++;
            left_sum+=a[j];
        }
        right_sum+=a[i];
        dp[i]=a[i]+*mst.begin();
        mst.insert(dp[i]);
    }
    int mn=1e14,sum=0;
    for(int i=n;i>0;i--){
        if(sum<=mid){
            mn=min(mn,dp[i]);
            sum+=a[i];
        }
        else{
            break;
        }
    }
    if(mn<=mid){
        ans=mid;
        hi=mid-1;
    }
    else{
        lo=mid+1;
    }
}
cout<<ans;
```

## Problem 2 (DP + Segment Tree)

Given an array of numbers $a_1, a_2, ... a_n$ ($1 \leq n \leq 5 \times 10^5, 1 \leq a_i \leq 5 \times 10^5$). Find maximum length of subsequence of $a$ such that absolute difference between any two adjacent terms is at most $d$ ($0 \leq d \leq 5 \times 10^5$)

## Solution to Problem 2
```cpp
/*
Define dp[i] as maximum length subsequence satisfying given criteria and ending at index i.
Then dp[1] = 1 and dp[i] = 1 + max(dp[j] where j<i and |a[i]-a[j]| <= d). To find such a j
quickly, we can use a segment tree
*/

struct segtree{
    int sz;
    vector<int>values;

    int NEUTRAL_ELEMENT=0;

    int merge(int& a,int& b){
        return max(a,b);
    }

    int single(int v){
        
    }
    
    void init(int n){
        sz=1; 
        while(sz<n) sz<<=1; 
        values.resize(2*sz);
    }
    
    void build(vector<int>& a,int x,int lx,int rx){
        if(rx==lx+1){
            if(lx<(int)a.size()) values[x]=single(a[lx]); 
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
        if(i<m) set(i,v,2*x+1,lx,m); 
        else set(i,v,2*x+2,m,rx); 
        values[x]=merge(values[2*x+1],values[2*x+2]);
    }

    void set(int i,int v){
        set(i,v,0,0,sz);
    }
    
    int calc(int l,int r,int x,int lx,int rx){
        if(lx>=r || rx<=l) return NEUTRAL_ELEMENT; 
        if(lx>=l && rx<=r) return values[x]; 
        int m=lx+(rx-lx)/2; 
        int x1=calc(l,r,2*x+1,lx,m); 
        int x2=calc(l,r,2*x+2,m,rx); 
        return merge(x1,x2);
    }

    int calc(int l,int r){
        return calc(l,r,0,0,sz);
    }
};

vector<int>dp(n+1),mx(500001,0);
dp[1]=1;
int ans=1;
segtree st;
st.init(500000);
st.build(mx);
st.set(a[1],1);
for(int i=2;i<=n;i++){
    int l=max(0,a[i]-d);
    int r=min(500000,a[i]+d)+1;
    dp[i]=1+st.calc(l,r);
    st.set(a[i],dp[i]);
    ans=max(ans,dp[i]);
}
cout<<ans<<'\n';
```

## Problem 3 (Dijkstra's Algorithm)

You are playing a game consisting of $n$ stages, numbered $1, 2, ..., n$. Initially, only stage $1$ can be played. For each stage $i$ ($1 \leq i \leq n - 1$) that can be played, you can perform one of the following two actions at stage $i$:
- Spend $a_i$ seconds to clear stage $i$. This allows you to play stage $i+1$
- Spend $b_i$ seconds to clear stage $i$. This allows you to play stage $x_i$\
\
Calculate minimum time to play stage $n$

## Solution to Problem 3
```cpp
/*
We build a directed graph consisting of n nodes. A directed edge from node i to j
with weight w means that you can play stage j after clearing stage i in w seconds.
We just need to find minimum distance of node n from node 1
*/

vector<vector<pair<int,int>>>g(n+1);
for(int i=1;i<n;i++){
    g[i].push_back(make_pair(i+1,a[i]));
    g[i].push_back(make_pair(x[i],b[i]));
}
priority_queue<pair<int,int>,vector<pair<int,int>>,greater<pair<int,int>>>pq;
vector<int>dist(n+1,INT_MAX);
dist[1]=0;
pq.push(make_pair(0,1));
while(!pq.empty()){
    int u=pq.top().second;
    int d_u=pq.top().first;
    pq.pop();
    if(d_u!=dist[u]){
        continue;
    }
    for(auto v:g[u]){
        if(d_u+v.S<dist[v.first]){
            dist[v.first]=d_u+v.second;
            pq.push(make_pair(dist[v.first],v.first));
        }
    }
}
cout<<dist[n];
```

## Problem 4 (Dijkstra's Algorithm)

There are $n$ railway stations: station $1$, station $2$, ..., station $n$. $m$ trains operate between the stations, with description represented as tuple of $6$ positive integers ($l_i, d_i, k_i, c_i, u_i, v_i$). This corresponds to the information that the train departs from station $u_i$ at times ($l_i, l_i + d_i, ..., l_i + (k_i - 1)d_i$) and arrives at station $v_i$ after $c_i$ time from departure. For each of the stations $1$ to $n-1$, tell the latest time when a person can reach that station so that he can catch trains to reach station $n$ or determine that it is unreachable

## Solution to Problem 4
```cpp
/*
We shall make a weighted directed graph where an edge of weight c directed from
node u to node v means that the train departs station v for station u and reaches
in time c. We shall then run Dijkstra's Algorithm on this graph to determine the
answers
*/

vector<vector<vector<int>>>g(n+1);
for(int i=0;i<m;i++){
    g[v[i]].push_back({u[i],l[i],d[i],k[i],c[i]});
}
priority_queue<pair<int,int>>pq;
vector<int>dist(n,-1);
for(auto x:g[n]){
    int v=x[0],l=x[1],d=x[2],k=x[3],c=x[4];
    dist[v]=l+(k-1)*d;
    pq.push(make_pair(dist[v],v));
}
while(!pq.empty()){
    int u=pq.top().second,d_u=pq.top().first;
    pq.pop();
    if(d_u!=dist[u]){
        continue;
    }
    for(auto x:g[u]){
        int v=x[0],l=x[1],d=x[2],k=x[3],c=x[4];
        int t=d_u-c;
        if(t<l){
            continue;
        }
        int p=min(k,1+(t-l)/d);
        int d_v=l+(p-1)*d;
        if(t>=d_v && d_v>=dist[v]){
            dist[v]=d_v;
            pq.push(make_pair(d_v,v));
        }
    }
}
for(int i=1;i<n;i++){
    if(dist[i]<0){
        cout<<"Unreachable ";
    }
    else{
        cout<<dist[i]<<" ";
    }
}
```
