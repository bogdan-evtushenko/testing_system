#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
#include <ext/rope>
 
#define ll long long
#define ll128 __uint128_t
#define ld long double
#define vll vector <ll>
#define vvll vector <vll>
#define pll pair <ll, ll>
 
#define rep(i, a, b) for(ll i = a; i < b; i++)
#define per(i, a, b) for(ll i = a - 1; i >= b; --i)
 
#define endl "\n"
#define pb push_back
#define pf push_front
    
#define all(v) (v).begin(), (v).end()
#define rall(v) (v).rbegin(), (v).rend()
 
#define sorta(v) sort(all(v))
#define sortd(v) sort(rall(v))
#define vld vector<ld>
 
#define debug if (1)
#define log(val) debug {cout << "\n" << #val << ": " << val << "\n";}
 
#define ios ios_base::sync_with_stdio(0); cin.tie(0); cout.tie(0);
 
#define mod (ll)(1e9 + 7)
 
using namespace std;
using namespace __gnu_cxx;
using namespace __gnu_pbds;
 
ostream & operator << (ostream & out, vll & a) {
    for(auto i : a) out << i << " ";
    return out;
}
 
istream & operator >> (istream & in, vll & a) {
    for(auto &i : a) in >> i;
    return in;
}

vll g[200010];

vector<pll> curs;

void dfs(ll v, ll p = -1, ll curh = 0) {
    if(g[v].size() == 1) {
        curs.pb({curh, v});
    }
    for (auto i : g[v]) {
        if (i != p) {
            dfs(i, v);
        }
    }
}

ll v1, v2, v3;

map<pll, ll> mp;

pll edge(ll a, ll b) {
    return {min(a, b), max(a, b)};
}

ll dfsuse(ll v, ll p = -1) {
    for (auto i : g[v]) {
        if (i != p) dfsuse(i, v);
    }
    if (v1 == v || v2 == v) {
        mp[edge(v, p)] = 1;
    } 
}

ll ans = 0;

ll dfsmx(ll v, ll p = -1, ll curmx = 0) {
    if (curmx > ans) {
        ans = curmx;
        v3 = v;
    }
    for (auto i : g[v]) {
        if (i != p) {
            dfsmx(i, v, curmx + !mp[edge(i, v)]);
        }
    }
}

int main() {
    // freopen("rmq.in", "r", stdin);
    // freopen("rmq.out", "w", stdout);
    // ios;

    ll n;
    cin >> n;
    rep(i, 0, n - 1) {
        ll a, b;
        cin >> a >> b;
        g[a - 1].pb(b - 1);
        g[b - 1].pb(a - 1);
    }

    dfs(0);

    sort(all(curs));

    v1 = curs[0].second;
    v2 = curs[1].second;
    dfsuse(0);
    dfsmx(0);
    cout << v1 + 1 <<  " " << v2 + 1 << " " << v3 + 1 << endl;
    
    return 0;
}
