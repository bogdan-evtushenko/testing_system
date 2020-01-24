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

int main() {
    // freopen("rmq.in", "r", stdin);
    // freopen("rmq.out", "w", stdout);
    ios;

    ll n, h;
    cin >> n >> h;
    vll a(n);
    cin >> a;
    vll pref(n, 0);
    pref[0] = a[0];
    rep(i, 1, n) {
        pref[i] = pref[i - 1] + a[i];
    }
    ll last = n - 1;
    ll ans = 1e18;
    ll cnt = 1;
    ll oo = 0;    

    ll dbg = 1;
    ll pos = n - 1;
    ll sum = h * (h + 1) / 2;
    ll curcnt = 0;
    for(ll i = n - 1; i - h + 1 >= 0; i--) {
        cnt = h;
        curcnt++;
        ll oo = 0;
        cnt -= (i - pos);
        per(j, pos + 1, i - h + 1) {  
            curcnt++;  
            if (a[j] > cnt) {
                ll curh = min(h, a[j]);
                i -= curh - cnt;
                if (a[j] > h) {
                    i--;
                }
                oo = 1;
                break;
            }
            cnt--;
        }

        if(!oo) {
            ll res = pref[i] - (i - h >= 0 ? pref[i - h] : 0);
            ans = min(ans, sum - res);
            pos = i - h;
        } else {
            pos = i;
            i++;
        }
    }
    if(curcnt > 3 * n) {
        assert(false);
    }
    cout << curcnt << endl;
    if(ans == 1e18) cout << -1 << endl;
    else cout << ans << endl;
    
    return 0;
}
