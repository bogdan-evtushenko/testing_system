This ideone code will use for teamNotebook in future

Defines, includes and namespaces(begin cpp file):

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

Algorithms, data structure

For adding algorithm or data structure use this form please:

////////////////////////////////////////////////
@yourNickname
Some text about pasted code

Code
////////////////////////////////////////////////

////////////////////////////////////////////////
@Bogdan
SegmentTree

struct Node {
    ll val;

    Node() : val(0LL) {}
    Node(ll value) : val(value) {}
};

const ll SZ = 2e5;
ll a[SZ];
Node t[4 * SZ];
ll add[4 * SZ];

struct SegTree {

    ll n;
    ll lazyOp;
    // 1 - if sum lazyOp
    // 2 - if max, min lazyOp

    ll passiveEl = 0LL; // getOp(a, passiveEl) = a
    ll neitralEl = 0LL; // if(b != neitralEl) a = b;(for lazy)

    SegTree(ll sz, ll lazy) {
        n = sz; lazyOp = lazy;
    }

    void input() {
        rep(i, 0, n) {
            cin >> a[i];
        }
    }

    Node getOp(Node left, Node right) {
        return Node(max(left.val, right.val));
    }

    void pull(ll v) {
        t[v] = getOp(t[v * 2 + 1], t[v * 2 + 2]);
    }

    void build(ll v, ll l, ll r) {
        if(l == r) {
            t[v] = Node(a[l]);
            return;
        }

        ll m = (l + r) / 2;
        build(v * 2 + 1, l, m);
        build(v * 2 + 2, m + 1, r);

        pull(v);
    }

    //////////////////////////////////////
    //Lazy operation
    ll pullChildOp(ll val, ll segSize) {
        if(lazyOp == 1) return val * segSize;
        else if(lazyOp == 2) return val;
        assert(false);
    }

    void push(ll v, ll l, ll m, ll r) {
        // Change += on =
        if(add[v] != neitralEl) {
            add[v * 2 + 1] += add[v];
            add[v * 2 + 2] += add[v];

            t[v * 2 + 1].val += pullChildOp(add[v], m - l + 1);
            t[v * 2 + 2].val += pullChildOp(add[v], r - m);

            add[v] = neitralEl;
        }
    }
    //////////////////////////////////////

    void updateSeg(ll v, ll tl, ll tr, ll l, ll r, ll val) {
        if(l > r) return;
        if(l == tl && r == tr) {
            // Change += on =
            t[v].val += pullChildOp(val, r - l + 1);
            add[v] += val;
            return;
        }

        ll tm = (tl + tr) / 2;

        push(v, tl, tm, tr);

        updateSeg(v * 2 + 1, tl, tm, l, min(tm, r), val);
        updateSeg(v * 2 + 2, tm + 1, tr, max(tm + 1, l), r, val);

        pull(v);
    }

    void updateEl(ll v, ll tl, ll tr, ll pos, Node val) {
        if(tl == tr) {
            t[v] = val;
            return;
        }

        ll tm = (tl + tr) / 2;

        if(pos <= tm) {
            updateEl(v * 2 + 1, tl, tm, pos, val);
        } else {
            updateEl(v * 2 + 2, tm + 1, tr, pos, val);
        }

        pull(v);
    }

    Node get(ll v, ll tl, ll tr, ll l, ll r) {

        if(l > r) return passiveEl;
        if(tl == l && r == tr) return t[v];

        ll tm = (tl + tr) / 2;

        if(lazyOp) push(v, tl, tm, tr);
        return getOp(get(v * 2 + 1, tl, tm, l, min(tm, r)),
                        get(v * 2 + 2, tm + 1, tr, max(tm + 1, l), r));
    }

    //Simply operations
    void updateSeg(ll l, ll r, ll val) {
        assert(lazyOp);
        updateSeg(0, 0, n - 1, l, r, val);
    }

    void updateEl(ll pos, ll val) {
        updateEl(0, 0, n - 1, pos, val);
    }

    Node get(ll l, ll r) {
        return get(0, 0, n - 1, l, r);
    }

    void build() {
        build(0, 0, n - 1);
    }

};

////////////////////////////////////////////////




////////////////////////////////////////////////
@Bogdan
Cartesian Tree in array

struct Node {

    ll priority, cnt, val, min;
    bool rev;
    Node * l, * r, *p;
    Node(ll n) : priority(rand()), cnt(1), val(n), min(n), rev(0), l(nullptr), r(nullptr), p(nullptr) {};

};

void buildTree(Node *& t, string & s) {

    Node * last = new Node((ll)(s[0] - '0'));;
    rep(i, 1, s.size()) {

        auto cur = new Node((ll)(s[i] - '0'));
        if(cur -> priority < last -> priority) {
            last -> r = cur;
            cur -> p = last;
            last = cur;
        } else {
            while(last -> p && last -> priority < cur -> priority) {
                last = last -> p;
            }
            if(last -> priority < cur -> priority) {
                cur -> l = last;
                last -> p = cur;
                last = cur;
            } else {
                cur -> l = last -> r;
                cur -> p = last;
                last -> r = cur;
                last = cur;
            }
        }
    }

    while(last -> p) {
        last = last -> p;
    }
    t = last;
}

ll cnt(Node * t) {
    if(!t) return 0;
    return t -> cnt;
}

ll minNode(Node * t) {
    if(!t) return 2e9;
    return t -> min;
}

void update(Node * t) {
    if(!t) return;
    t -> cnt = 1 + cnt(t -> l) + cnt(t -> r);
    t -> min = min({t -> val, minNode(t -> l), minNode(t -> r)});
}

void push(Node * t) {
    if(t && t -> rev) {
        t -> rev = 0;
        if(t -> l) t -> l -> rev ^= 1;
        if(t -> r) t -> r -> rev ^= 1;
        swap(t -> l, t -> r);
    }
}

void print(Node * t) {
    if(!t) return;
    print(t -> l);
    cout << t -> val << " " << t -> min << " " << t -> cnt << "   ";
    print(t -> r);
}

void split(Node * t, Node *& l, Node *& r, ll pos) {
    push(t);
    if(!t) {
		l = r = nullptr;
		update(l);
		update(r);
	}
    else {
        if(pos <= cnt(t -> l)) {
            split(t -> l, l, t -> l, pos);
            r = t;
            update(r);
        } else {
            split(t -> r, t -> r, r, pos - cnt(t -> l) - 1);
            l = t;
            update(l);
        }
    }
}

void merge(Node * l, Node * r, Node *& t) {
    push(l);
    push(r);
    if(!l || !r) {
        t = l ? l : r;
    } else {
        if(l -> priority > r -> priority) {
            merge(l -> r, r, l -> r);
            t = l;
        } else {
            merge(l, r -> l, r -> l);
            t = r;
        }
    }
    update(t);
}

void insert(Node *& t, Node * cur, ll pos) {
    Node * l, * r;
    split(t, l, r, pos);
    merge(l, cur, t);
    merge(t, r, t);
}

ll getMin(Node * t, ll l, ll r) {
    Node * l1, *r1, *r2;
    split(t, l1, r1, r);
    split(l1, l1, r2, l - 1);
    ll ans = minNode(r2);
    merge(l1, r2, t);
    merge(t, r1, t);
    return ans;
}

void rev(Node *& t, ll l, ll r) {
    Node * l1, * r1, * r2;
    split(t, l1, r1, r);
    split(l1, l1, r2, l - 1);
    if(r2) r2 -> rev ^= 1;
    merge(l1, r2, t);
    merge(t, r1, t);
}
////////////////////////////////////////////////

////////////////////////////////////////////////
@Bogdan
Cartesian Tree(ordinary)

void insert (pitem &t, pitem it)
{
  if (!t)
    t = it;
  else if (it->Priority > t->Priority)
    split (t, it->Key, it->l, it->r),  t = it;
  else
    insert (it->Key < t->Key ? t->l : t->r, it);
}

void erase (pitem &t, int Key)
{
  if (t->Key == Key)
    merge (t->l, t->r, t);
  else
    erase (Key < t->Key ? t->l : t->r, Key);
}
////////////////////////////////////////////////

////////////////////////////////////////////////
@Bogdan
Kun algorithm maximal matching
bool dfs(ll v) {
    if(used[v]) return 0;
    used[v] = 1;
    for(auto i : g[v]) {
        if(!rm[i] || dfs(rm[i] - 1)) {
            rm[i] = v + 1;
            return 1;
        }
    }
    return 0;
}

rep(i, 0, n) {
    	used.assign(n, 0);
    	dfs(i);
}
////////////////////////////////////////////////

////////////////////////////////////////////////
@Bogdan
Graphs find bridges
void dfs(int v, int p){
    s[v] = up[v] = timer++;
    for(auto vertex : g[v]){
        if(s[vertex] == 0){
            dfs(vertex, v);
            up[v] = min(up[v], up[vertex]);
            if(up[vertex] > s[v]){
                bridges.insert(mp[Edge(v, vertex)]);
            }
        }
        else {
            up[v] = min(up[v], s[vertex]);
        }
    }
}
////////////////////////////////////////////////

////////////////////////////////////////////////
@Bogdan
Graphs find cut vertices

void dfs(int v, int p){
    int cnt = 0;
    s[v] = up[v] = timer ++;
    for(auto vertex : g[v]){
        if(s[vertex] == 0){
            cnt ++;
            dfs(vertex, v);
            up[v] = min(up[v], up[vertex]);
            if(up[vertex] >= s[v] && p != -1){
                artPoints.insert(v);
            }
        }
        else if(vertex != p){
            up[v] = min(up[v], s[vertex]);
        }
    }
    if(p == -1 && cnt > 1){
        artPoints.insert(v);
    }
}
////////////////////////////////////////////////

////////////////////////////////////////////////
@Bogdan
LCA

void dfs(ll v = 0, ll p = 0) {

    tin[v] = timer++;

    up[v][0] = p;
    rep(i, 1, l + 1) {
        up[v][i] = up[up[v][i - 1]][i - 1];
    }
    for(auto i : g[v]) {
        if(i != p) {
            dist[i] = dist[v] + 1;
            dfs(i, v);
        }
    }
    tout[v] = timer++;
}

bool parent(ll a, ll b) {
    return (tin[a] <= tin[b] && tout[a] >= tout[b]);
}

ll lca(ll a, ll b) {
    if(parent(a, b)) return a;
    if(parent(b, a)) return b;
    for(int i = l; i >= 0; i--) {
        if(!parent(up[a][i], b)) a = up[a][i];
    }
    return up[a][0];
}
////////////////////////////////////////////////

////////////////////////////////////////////////
@Bogdan
Bor

void add(string & t) {
    ll cur = 0;
    rep(i, 0, t.size()) {
        if(cur) bor[cur].cnt++;
        if(bor[cur].next[t[i]] == 0) {
            bor.pb({0});
            bor[cur].next[t[i]] = bor.size() - 1;
            cur = bor.size() - 1;
        } else {
            cur = bor[cur].next[t[i]];
        }
    }
}
////////////////////////////////////////////////

////////////////////////////////////////////////
@Bogdan

long long prime(long long a){

    long long i;

    if (a==2)
        return 1;
    if(a<=1 || a%2==0)
        return 0;

   for(i=3;i*i<=a;i+=2){
        if(a%i==0)
            return 0;
   }

    return 1;

}
////////////////////////////////////////////////

////////////////////////////////////////////////
@Bogdan
Improve Eratosfen sieve

const int N = 100;
int lp[N+1];
vector<int> pr;

for (int i=2; i<=N; ++i) {
    if (lp[i] == 0) {
        lp[i] = i;
        pr.push_back (i);
    }
for (int j=0; j<(int)pr.size() && pr[j]<=lp[i] && i*pr[j]<=N; ++j)
    lp[i * pr[j]] = pr[j];
}
////////////////////////////////////////////////

////////////////////////////////////////////////
@Bogdan

long long gcd(long long a , long long b){
    while(b!=0){
        long long t = b;
        b = a % b;
        a = t;
    }
    return a;
}

long long lcm (long long a, long long b) {
	return a / gcd (a, b) * b;
}

int C (int n, int k) {
	double res = 1;
	for (int i=1; i<=k; ++i)
		res = res * (n-k+i) / i;
	return (int) (res + 0.01);
}

int CnkTr(){
    const int maxn = ...;
    int C[maxn+1][maxn+1];
    for (int n=0; n<=maxn; ++n) {
	C[n][0] = C[n][n] = 1;
	for (int k=1; k<n; ++k)
		C[n][k] = C[n-1][k-1] + C[n-1][k];
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
@Bogdan
Рюкзак:

for(int i=0;i<=w;i++){
    d[0][i]=0;
}
for(int i=1;i<=m;i++){
    for(int j=1;j<=w;j++){
            if(j-a[i]<0){
                d[i][j]=d[i-1][j];
            }
            else
        d[i][j]=max(a[i]+d[i-1][j-a[i]],d[i-1][j]);
    }
}
cout<<d[m][w];
////////////////////////////////////////////////

////////////////////////////////////////////////
@Bogdan
bin search

while(l + 1 < r) {
    ll m = (l + r) / 2;
    if(can(m) < n) {
        r = m;
    } else {
        l = m;
    }
}
////////////////////////////////////////////////

////////////////////////////////////////////////
@Bogdan

ll binpow (ll a, ll n, ll m) {
	ll res = 1;
	while (n) {
		if (n & 1)
			res = res * a % m;
		a = a * a % m;
		n >>= 1;
	}
	return res;
}
////////////////////////////////////////////////

////////////////////////////////////////////////
@Bogdan

bool cmp(string findStr, string str) {
    ll n = findStr.size();
    findStr += "%" + str;
    vector<ll> p(findStr.size());
    ll k = 0;
    p[0] = 0;
    rep(i, 1, findStr.size()) {
        while(k > 0 && findStr[k] != findStr[i]) {
            k = p[k - 1];
        }
        if(findStr[k] == findStr[i]) {
            k++;
        }
        p[i] = k;
        if(p[i] == n) {
            //cout<<i - 2 * n<< " ";
            return 1;
        }
    }
    return 0;
}
////////////////////////////////////////////////

////////////////////////////////////////////////
@Bogdan
Non - recursive dfs

void dfs(ll v) {

    vector<pll> st;
    st.pb({v, -1});
    while(!st.empty()) {
        auto & cur = st.back();
        if(cur.second == -1) {
            res.pb(cur.first);
            used[cur.first] = 1;
        }
        ll in = cur.second + 1;
        cur.second = in;
        if(in < g[cur.first].size()) {
            ll i = g[cur.first][in];
            if(!used[i]) {
                st.pb({i, -1});
                continue;
            }
        } else {
            st.pop_back();
        }
    }
}
////////////////////////////////////////////////

////////////////////////////////////////////////
@Bogdan
Fast read

inline ll read()
{
 char c=getchar();
 ll x=0,f=1;
 while(c>'9'||c<'0')
 {
  if(c=='-') f=-1;
  c=getchar();
 }
 while(c>='0'&&c<='9')
 {
  x=x*10+c-'0';
  c=getchar();
 }
 return x*f;
}
////////////////////////////////////////////////

////////////////////////////////////////////////
@Bogdan
Queue with push_back, pop_front and find Min/Max O(1) complexity(average)

struct myQueue {
    stack<pll> s1, s2;

    int size() {
        return s1.size() + s2.size();
    }

    bool isEmpty() {
        return size() == 0;
    }

    long long getMax() {
        if (isEmpty()) {
            return -2e9;
        }
        if (!s1.empty() && !s2.empty()) {
            return gcd(s1.top().second, s2.top().second);
        }
        if (!s1.empty()) {
            return s1.top().second;
        }
        return s2.top().second;
    }

    void push(long long val) {
        if (s2.empty()) {
            s2.push({val, val});
        } else {
            s2.push({val, gcd(val, s2.top().second)});
        }
    }

    void pop() {
        if (s1.empty()) {
            while (!s2.empty()) {
                if (s1.empty()) {
                    s1.push({s2.top().first, s2.top().first});
                } else {
                    s1.push({s2.top().first, gcd(s2.top().first, s1.top().second)});
                }
                s2.pop();
            }
        }
        assert(!s1.empty());
        s1.pop();
    }
};
////////////////////////////////////////////////

Geometry

////////////////////////////////////////////////
@Bogdan
Psevdo multiply vectors

bool psevdo(pll a, pll b, pll c) {
    //return c under line ab or not
    return (((b.first - a.first) * (c.second - a.second) - (b.second - a.second) * (c.first - a.first)) >= 0);
}
////////////////////////////////////////////////

////////////////////////////////////////////////
@Bogdan
Intersection line and circle

pair<ld, ld> intersection(double x, double y, double x1, double y1, double xk, double yk, double r) {
    double a1 = (y1 - y), b = (x - x1), c = -(x * (y1 - y) + y*(x - x1));
    double c1 = a1 * xk + b * yk + c;
    double x0 = xk - a1 * c1 / (a1 * a1 + b * b), y0 = yk - b * c1 / (a1 * a1 + b * b);
    if(c1 * c1 > r * r * (a1 * a1 + b * b)){
        return {2e9, 2e9};
    }
    else if(c1 * c1 == r * r * (a1 * a1 + b * b)){
        return {x0, y0};
    }
    else {
        double d = r*r - c1*c1/(a1*a1+b*b);
        double mult = sqrt (d / (a1*a1+b*b));
        double ax,ay,bx,by;
        ax = x0 + b * mult;
        bx = x0 - b * mult;
        ay = y0 - a1 * mult;
        by = y0 + a1 * mult;
        pair<ld, ld> cur1 = {ay, ax};
        pair<ld, ld> cur2 = {by, bx};
        pair<ld, ld> cur = min(cur1, cur2);
        if(cur.first == ay && cur.second == ax) {
            return {ax, ay};
        } else {
            return {bx, by};
        }
    }
}
////////////////////////////////////////////////

////////////////////////////////////////////////
@Bogdan
Intersect two segments + (a, b, c in line)

bool peret(int x1,int y1,int x2,int y2,int  x3,int y3,int x4,int y4){
   	double x ,y;
	int a=(y2-y1),b=(x1-x2),c=(x1*(y2-y1)+y1*(x1-x2)),a1=(y4-y3),b1=(x3-x4),c1=(x3*(y4-y3)+y3*(x3-x4));
	if((a*b1-a1*b)==0){
		return (intersect_1 (x1,x2,x3,x4) && intersect_1 (y1, y2,y3,y4));
	}
	else{
		x=(c1*b-b1*c)*1./(b*a1-b1*a)*1.;
		y=(c*a1-c1*a)*1./(b*a1-b1*a)*1.;
		if((min(x1,x2)<=x && x<=max(x1,x2)) && (min(y1,y2)<=y && y<=max(y1,y2))
		&& (min(x3,x4)<=x && x<=max(x3,x4)) && (min(y3,y4)<=y && y <=max(y3,y4))){
			return 1;
		}
	    	return 0;
	}
}
////////////////////////////////////////////////

////////////////////////////////////////////////
@Bogdan

Turn point 

pair<ld, ld> turn(ld x, ld y, ld ug) {
    ld sn = sin(ug);
    ld cs = cos(ug);
 
    ld curx = mainx + (x - mainx) * cs - (y - mainy) * sn;
    ld cury = mainy + (x - mainx) * sn + (y - mainy) * cs;
    return {curx, cury};
}
////////////////////////////////////////////////

////////////////////////////////////////////////
@Dan
//Sparse Table

const ll N = 1e5 + 5;
const ll logN = log2(N);
 
ll mas[N];
 
ll table[logN + 1][N];
ll pow2[logN + 1];
 
void build() {
    pow2[0] = 1;
 
    rep(i, 1, logN + 1) {
        pow2[i] = pow2[i - 1];
        pow2[i] *= 2;
    }
 
    rep(i, 0, N)
       	table[0][i] = mas[i];
 
    ll len = 2, power = 1, l = 0;
 
    while (power <= logN) {
        while (l + len - 1 < N) {
            table[power][l] = min(table[power - 1][l], table[power - 1][l + len / 2]);
            l++;
        }
 
        power++;
        len = pow2[power];
        l = 0;
    }
}
 
ll query(ll l, ll r) {
    if (l > r)
        swap(l, r);
        
    ll power = log2(r - l + 1);
 
    return min(table[power][l], table[power][r - pow2[power] + 1]);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
@Dan
// LIS

const ll N = 1e5 + 5;

ll mas[N], p[N], ind[N];

    vll dp(n + 1, mod);

    dp[0] = -mod;

    ll mx = 0;

    rep(i, 0, n) {
        ll j = upper_bound(all(dp), mas[i]) - dp.begin();

        if (dp[j - 1] < mas[i]) {
            dp[j] = mas[i];
            mx = max(mx, j);

            p[i] = ind[j - 1];
            ind[j] = i;
        }
    }

    ll curInd = ind[mx];

    deque <ll> d;

    rep(i, 0, mx) {
        d.pf(mas[curInd]);

        curInd = p[curInd];
    }

    cout << d.size() << "\n";

    for (auto c : d)
        cout << c << " ";
    
/////////////////////////////////////////////////////////////////////
@Dan
Hamilton cycle

const ll N = 21;

ll dp[(1 << N)][N];

rep(i, 0, n)
    rep(mask, 0, (1 << n))
        dp[mask][i] = mod;

dp[1][0] = 0;

rep(mask, 0, (1 << n))
    rep(i, 1, n)
        if ((mask >> i) & 1)
            rep(j, 0, n)
                if (i != j && (mask >> j) & 1)
                    dp[mask][i] = min(dp[mask][i], dp[mask ^ (1 << i)][j] + dst[i][j]);

ll mn = mod;

rep(i, 1, n)
    mn = min(mn, dp[(1 << n) - 1][i] + dst[0][i]);

return mn;
/////////////////////////////////////////////////////////////////////////////
@Dan
Дейкстра

const ll N = 1e3 + 5;

struct edge {
    ll v, dst;  
};
 
const bool operator < (const edge &a, const edge &b) {
    return a.dst > b.dst;
}

vector <edge> graph[N];

	vll dst(n, mod);
 
    priority_queue <edge> q;
 
    q.push({beg, 0});
 
    dst[beg] = 0;
 
    while (!q.empty()) {
        edge cur = q.top();
 
        q.pop();
 
        if (cur.dst > dst[cur.v])
            continue;
 
        for (auto c : graph[cur.v]) {
            if (c.dst + cur.dst < dst[c.v]) {
                dst[c.v] = c.dst + cur.dst;
 
                q.push({c.v, dst[c.v]});
            }
        }
    }
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
@Dan
Fenwick

//one dimentional

void add(ll in, ll val) {
    for (ll i = in; i < N; i |= (i + 1))
        mas[i] += val;
}

ll sum(ll in) {
    ll s = 0;

    for (ll i = in; i >= 0; i = (i & (i + 1)) - 1)
        s += mas[i];

    return s;
}

// two dimentional

void add(ll x, ll y, ll val) {
    for (ll i = x; i < N; i |= (i + 1))
        for (ll j = y; j < N; j |= (j + 1))
            mas[i][j] += val;
}

ll sum(ll x, ll y) {
    ll s = 0;

    for (ll i = x; i >= 0; i = (i & (i + 1)) - 1)
        for (ll j = y; j >= 0; j = (j & (j + 1)) - 1)
            s += mas[i][j];

    return s;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
@Dan
// binpow and binmul

ll binmul(ll a, ll b, ll m) {   
    ll res = 0;

    while (b) {
        if (b & 1)
            res = (res + a) % m;

        a <<= 1;

        b >>= 1;

        a %= m;
    }

    return res;
}

ll binpow(ll a, ll b, ll m) {   
    ll res = 1;

    while (b) {
        if (b & 1)
            res = mul(res, a, m);

        a = mul(a, a, m);

        b >>= 1;
    }

    return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
@Bogdan

Euler path

void dfs(ll v) {
    while(!g[v].empty()) {
        ll cur = g[v].back();
        g[v].pop_back();
        dfs(cur);
    }
    
    ans.pb(getS(v));
}
////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////
@Bogdan
// z - function

vll zFunction (string & s) {
	ll n = s.size();
	vll z(n);
	ll l = 0, r = 0;
	rep(i, 1, n) {
		if (i <= r)
			z[i] = min (r - i + 1, z[i - l]);
		while (i + z[i] < n && s[z[i]] == s[i + z[i]])
			++z[i];
		if (i + z[i] - 1 > r)
			l = i,  r = i + z[i] - 1;
	}
	return z;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////
@Dan
// Centroid decomposition

const ll N = 5e4 + 5;

vvll graph;

vll sz, vec;

ll cnt = 0, curSz = 0, k, n;

bool used[N];

map <ll, ll> m;

void countSz(ll a, ll p = -1) {
    sz[a] = 1;

    for (auto c : graph[a])
        if (c != p && !used[c]) {
            countSz(c, a);

            sz[a] += sz[c];
        }
}

ll findCenter(ll a, ll p = -1) {
    for (auto c : graph[a])
        if (!used[c] && c != p && sz[c] > curSz / 2)
            return findCenter(c, a);

    return a;
}

void count(ll a, ll h, ll p = -1) {

    vec.pb(h);

    cnt += m[k - h];
    //cout << "k - h" << " " << m[k - h] << "\n";

    for (auto c : graph[a])
        if (!used[c] && c != p)
            count(c, h + 1, a);
}

void rec(ll a) {
    m.clear();

    m[0] = 1;

    countSz(a);

    curSz = sz[a];

    ll center = findCenter(a);

    used[center] = 1;

    for (auto c : graph[center])
        if (!used[c]) {
            vec.clear();
            count(c, 1);

            for (auto cc : vec)
                m[cc]++;
        }

    for (auto c : graph[center])
        if (!used[c])
            rec(c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
@Dan 
// DSU

const ll N = 2e5 + 5;

ll p[N], s[N];

void add(ll i) {
    p[i] = i;
    s[i] = 1;
}

ll get(ll v) {
    return p[v] == v ? v : p[v] = get(p[v]);
}

void unite(ll a, ll b) {
    a = get(a);
    b = get(b);

    if (a != b) {
        if (s[a] > s[b])
            swap(a, b);

        p[a] = b;
        s[b] += s[a];
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
@Dan
// Knapsack with recovery
// 1 - index

rep(i, 1, n + 1)
        rep(j, 0, allW + 1) {
            dp[i][j] = dp[i - 1][j];

            if (j - w[i] >= 0)
                dp[i][j] = max(dp[i][j], dp[i - 1][j - w[i]] + p[i]);
        }

    ll cur = n, curW = allW;

    deque <ll> ans;

    while (dp[cur][curW]) {
        if (dp[cur - 1][curW] == dp[cur][curW])
            cur--;
        else {
            curW -= w[cur];
            ans.pf(cur);
            cur--;
        }
    }

    cout << dp[n][curW] << "\n";
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
@Bohdan
// new binary search
// (binary lifting)

for (ll i = 30; i >= 0; i--) {
        cur3 += (1LL << i);
        if (!can(cur3)) {
            cur3 -= (1LL << i);
        }
}
//////////////////////////////////////////////////////////////////////////////////////////////////

Some useful tips from C++

rope<int> rp; (неявное ДД)

(faster than unordered_map<int, int>)
cc_hash_table<int, int> table;
gp_hash_table<int, int> table;

////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef tree<int, null_type, less<int>, rb_tree_tag, tree_order_statistics_node_update> ordered_set;
USAGE: 

ordered_set X;
X.insert(8);
X.insert(16);

cout<<*X.find_by_order(4)<<endl; // 16
cout<<(end(X)==X.find_by_order(6))<<endl; // true

cout<<X.order_of_key(-5)<<endl;  // 0
cout<<X.order_of_key(4)<<endl;   // 2
cout<<X.order_of_key(400)<<endl; // 5

////////////////////////////////////////////////////////////////////////////////////////////////////////

//cerr << "Time elapsed: " << clock() / (double)CLOCKS_PER_SEC << endl;
typedef __uint128_t ui128;
cout<< __builtin_popcount (4);


