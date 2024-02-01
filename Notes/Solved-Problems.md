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
