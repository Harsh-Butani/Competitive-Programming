## Problem 1 (Binary Search + DP + Two pointers)

Given an array of numbers $a_1, a_2, ... a_n$ ($1 \leq n \leq 10^5, 1 \leq a_i \leq 10^9$). Your task is to block some elements of the array in order to minimize its cost. Suppose you block the elements with
indices $1 \leq b_1 < b_2 ... < b_m \leq n$. The cost of the array is calculated as maximum of
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
