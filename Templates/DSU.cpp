#include<bits/stdc++.h>
using namespace std;

using ll = long long;
#define MOD 1000000007
#define TEST int t; cin>>t; while(t--)
#define F first
#define S second
#define PB push_back
#define MP make_pair
#define FAST_IO ios_base::sync_with_stdio(false); cin.tie(NULL);
#define FILE_IO freopen("file.in","r",stdin); freopen("file.out","w",stdout);

class DSU{
public:
    int n;
    vector<int>p,r;
    
    DSU(int _n){
        n=_n;
        p.resize(n+1,0);
        iota(p.begin(),p.end(),0);
        r.resize(n+1,0);
    }
    
    int find(int x){
        return p[x]=(x==p[x]?x:find(p[x]));
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
            }
            else{
                p[u]=v;
            }
        }
    }
};

void solve(){
    
}

int main(){
    FAST_IO
    TEST solve();
    return 0;
}
