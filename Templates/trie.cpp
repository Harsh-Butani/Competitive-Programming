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

struct node{
    struct node* children[26];
    bool end;
};

struct node* get_node(){
    struct node* n=new node;
    for(int i=0;i<26;i++){
        n->children[i]=nullptr;
    }
    n->end=false;
    return n;
}

void insert(struct node* root,string& s){
    struct node* temp=root;
    int n=s.length();
    for(int i=0;i<n;i++){
        int idx=s[i]-'a';
        if(!temp->children[idx]){
            temp->children[idx]=get_node();
        }
        temp=temp->children[idx];
    }
    temp->end=true;
}

bool search(struct node* root,string& s){
    struct node* temp=root;
    int n=s.length();
    for(int i=0;i<n;i++){
        int idx=s[i]-'a';
        if(!temp->children[idx]){
            return false;
        }
        temp=temp->children[idx];
    }
    return temp->end;
}

void solve(){
    
}

int main(){
    FAST_IO
    TEST solve();
    return 0;
}
