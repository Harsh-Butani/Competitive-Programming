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

struct item{

};

struct segtree{
    int sz;
    vector<item>values;

    item NEUTRAL_ELEMENT=;

    item merge(item a,item b){
        // Merged value of item a and item b
    }

    item single(int v){
        
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
    
    item calc(int l,int r,int x,int lx,int rx){
        if(lx>=r || rx<=l) return NEUTRAL_ELEMENT; 
        if(lx>=l && rx<=r) return values[x]; 
        int m=lx+(rx-lx)/2; 
        item x1=calc(l,r,2*x+1,lx,m); 
        item x2=calc(l,r,2*x+2,m,rx); 
        return merge(x1,x2);
    }

    item calc(int l,int r){
        return calc(l,r,0,0,sz);
    }
};

void solve(){
    
}

int main(){
    FAST_IO
    TEST solve();
    return 0;
}
