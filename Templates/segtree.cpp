#include<bits/stdc++.h>
using namespace std;

using ll = long long;
#define MOD 1000000007
#define TEST int t; cin>>t; while(t--)
#define F first
#define S second
#define PB push_back
#define MP make_pair
#define FAST_IO ios_base::sync_with_stdio(false); cin.tie(nullptr);
#define FILE_IO freopen("file.in","r",stdin); freopen("file.out","w",stdout);

struct segtree{
    ll sz;
    vector<ll>values;
    
    ll merge(ll a,ll b){
        return a+b;
    }
    
    void init(ll n){
        sz=1; 
        while(sz<n) sz<<=1; 
        values.assign(2*sz,0LL);
    }
    
    void build(vector<ll>& a,ll x,ll lx,ll rx){
        if(rx==lx+1){
            if(lx<(ll)a.size()) values[x]=a[lx]; 
            return;
        } 
        ll m=lx+(rx-lx)/2; 
        build(a,2*x+1,lx,m); 
        build(a,2*x+2,m,rx); 
        values[x]=merge(values[2*x+1],values[2*x+2]);
    }
    
    void set(ll i,ll v,ll x,ll lx,ll rx){
        if(rx==lx+1){
            values[x]=v; 
            return;
        } 
        ll m=lx+(rx-lx)/2; 
        if(i<m) set(i,v,2*x+1,lx,m); 
        else set(i,v,2*x+2,m,rx); 
        values[x]=merge(values[2*x+1],values[2*x+2]);
    }
    
    ll calc(ll l,ll r,ll x,ll lx,ll rx){
        if(lx>=r || rx<=l) return 0; 
        if(lx>=l && rx<=r) return values[x]; 
        ll m=lx+(rx-lx)/2; 
        ll s1=calc(l,r,2*x+1,lx,m); 
        ll s2=calc(l,r,2*x+2,m,rx); 
        return merge(s1,s2);
    }
};

void solve(){
    
}

int main(){
    FAST_IO
    TEST solve();
    return 0;
}
