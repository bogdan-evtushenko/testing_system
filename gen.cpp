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

int rand(int a, int b) {
    return a + rand() % (b - a + 1);
}

int main(int argc, char* argv[]) {
    srand(atoi(argv[1])); // atoi(s) converts an array of chars to int
   // ll n = rand() % cur + 1;
   // ll h = rand() % n + 1;

    puts("");
}