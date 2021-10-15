---
title: NamomoCamp-day1-dp
date: 2021-02-01 04:39:03
categories: [NamomoCamp]
tags:
mathjax: true
---

Solution for NamomoCamp2021-div2-day1-dp

<!-- more -->

# [New Year and Original Order (Good Bye 2017 G)](http://codeforces.com/contest/908/problem/G)

{% collapse Tag%}

数位dp

{% endcollapse %}

{% collapse Solution %}

用$f[d][i]$表示数字$d$在第$i$位上出现的次数，假设我们已知所有$f[d][i]$，则有
$$
ans=\sum_{d=0}^{9}\sum_{i=1}^{N}(f[d][i]\times10^{i-1}\times d)
$$
可惜的是$f[d][i]$未知，直接求$f[d][i]$也很难求

考虑下性质：`所有数字升序排序`，转化一下求$f[d][i]$的思路，用$g[d][i]$表示$\ge d$的数字出现$\ge i$次的次数，那么有$f[d][i]=g[d][i]-g[d+1][i]$

$g[d][i]$依旧不好直接求，但可以先求$h[d][i]$：$\ge d$的数字出现恰好$i$次的次数，$g[d][i]=\sum_{j=i}^{n}h[d][j]$

$h$就可以直接求了，数位dp一下即可，状态$h[pos][d][i][limit]$，状态数$O(10N^2)$

$k$表示当前位枚举的数字，转移式
$$
h[pos][d][i]=
\begin{cases}
h[pos+1][d][i]& k<d \\
h[pos+1][d][i-1]& k\ge d
\end{cases}
$$


{% endcollapse %}

{% collapse Code %}

```c++
#include <bits/stdc++.h>
using namespace std;

typedef pair<int, int> pi;
typedef long long ll;

const int maxn = 705;
const int inf = 0x3f3f3f3f;
const int mod = 1e9 + 7;

char s[maxn];
int a[maxn];
int n;

int f[10][maxn], w[maxn];
int g[10][maxn];
int h[maxn][10][maxn][2];

int dp(int pos, int d, int i, bool limit) {
    if (pos == n + 1) return i == 0 ? 1 : 0;
    if (h[pos][d][i][limit] != -1) return h[pos][d][i][limit];
    int up = limit ? a[pos] : 9;
    int ans = 0;
    for (int k = 0; k <= up; k++) {
        if (k >= d && i > 0) {
            ans += dp(pos + 1, d, i - 1, limit && k == up);
            ans %= mod;
        } else if (k < d) {
            ans += dp(pos + 1, d, i, limit && k == up);
            ans %= mod;
        }
    }
    return h[pos][d][i][limit] = ans;
}

int main() {

    int T = 1;
//    scanf("%d", &T);
    while (T--) {
        scanf("%s", s + 1);
        n = strlen(s + 1);
        for (int i = 1; i <= n; i++) {
            a[i] = s[i] - '0';
        }
        memset(h, -1, sizeof(h));
        for (int d = 0; d < 10; d++) {
            for (int i = 1; i <= n; i++) {
                for (int j = i; j <= n; j++) {
                    g[d][i] += dp(1, d, j, true);
                    g[d][i] %= mod;
                }
            }
        }
        for (int i = 1; i <= n; i++) {
            for (int d = 0; d < 9; d++) {
                f[d][i] = (g[d][i] - g[d + 1][i] + mod) % mod;
            }
            f[9][i] = g[9][i];
        }
        w[0] = 1;
        for (int i = 1; i < n; i++) {
            w[i] = w[i - 1] * 10ll % mod;
        }
        int ans = 0;
        for (int d = 0; d < 10; d++) {
            for (int i = 1; i <= n; i++) {
                ans += 1ll * d * f[d][i] * w[i - 1] % mod;
                ans %= mod;
            }
        }
        printf("%d\n", ans);
    }

    return 0;
}
```

{% endcollapse %}

<br>

# [Financiers Game (CF 729 F)](http://codeforces.com/contest/729/problem/F)

{% collapse Tag %}

dp，状态优化，记忆化搜索

{% endcollapse %}

{% collapse Solution %}

先来想一个暴力的状态，$f[i][j][k][m]$表示左边取$i$个，右边​取$j$个，上一步取了$k$，$m=0/1$表示轮到左/右边取，$f[i][j][k][m]$则表示该状态下的答案

转移显然，状态也记全了，但是这个空间是$N^3$的，开不下。

这题的要点在于注意到$k$的优化，来考虑一下能最大化$k$的情况，肯定是从$2$开始，每次都取$k+1$，即取$\{2,3,4,..,k\}$，此时有$\sum_{i=2}^{k}{i}=\frac{(k+2)\times(k-1)}{2} \leq N$，化简一下，可以得到一个大约的范围$k\leq \sqrt{2N} $

然后再以类似的想法来考虑$j$的优化，首先把$j$从`右边取j个`变成`左右已取的个数之差`，考虑最大化$j$，则应该是左边每次取$k+1$，而右边每次取$k$，那么某一时刻，可以表示为 左边取 $\{1,2,3,...,k-1,k\} $，右边取 $\{1,2,3,...,k-1\} $，类似上面算一下，可得 $k\leq \sqrt{N} $，因此 $-\sqrt{N}\leq j\leq \sqrt{N} $

优化$j$和$k$后空间其实已经够了，本题内存限制$512MB$，这样大概是$300MB$

但其实$i$也可以优化，最大化$i$的情况与最大化$j$的情况类似，结果大约是$i\leq \frac{N+\sqrt{N}}{2} $，这样就可以跑过$256MB$限制的情况了，过程自己算吧

{% endcollapse %}

{% collapse Code %}

```c++
#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef pair<int, int> pi;
typedef pair<ll, ll> pll;

const int maxn = 100005;
const int inf = 0x3f3f3f3f;
const ll INF = 0x3f3f3f3f3f3f3f3f;
const int mod = 1e9 + 7;

int a[4005];
int pre[4005], suf[4005];
int n;

int f[2100][131][91][2];
const int M = 65;

int dp(int i, int j, int k, int mode) {
    int B = i + M - j;
    if (mode == 0) {
        if (f[i][j][k][mode] != -inf) return f[i][j][k][mode];
        if (i + B + k <= n) {
            f[i][j][k][0] = max(f[i][j][k][0], dp(i + k, j + k, k, 1) + pre[i + k] - pre[i]);
        }
        if (i + B + k + 1 <= n) {
            f[i][j][k][0] = max(f[i][j][k][0], dp(i + k + 1, j + k + 1, k + 1, 1) + pre[i + k + 1] - pre[i]);
        }
        if (i + B + k > n) {
            f[i][j][k][0] = 0;
        }
    } else {
        if (f[i][j][k][mode] != inf) return f[i][j][k][mode];
        if (i + B + k <= n) {
            f[i][j][k][1] = min(f[i][j][k][1], dp(i, j - k, k, 0) - suf[n - B - k + 1] + suf[n - B + 1]);
        }
        if (i + B + k + 1 <= n) {
            f[i][j][k][1] = min(f[i][j][k][1], dp(i, j - k - 1, k + 1, 0) - suf[n - B - k] + suf[n - B + 1]);
        }
        if (i + B + k > n) {
            f[i][j][k][1] = 0;
        }
    }
    return f[i][j][k][mode];
}

int main() {

    scanf("%d", &n);
    for (int i = 1; i <= n; i++) {
        scanf("%d", a + i);
    }
    pre[0] = 0; suf[n + 1] = 0;
    for (int i = 1; i <= n; i++) pre[i] = pre[i - 1] + a[i];
    for (int i = n; i >= 1; i--) suf[i] = suf[i + 1] + a[i];

    for (int i = 0; i < 2100; i++) {
        for (int j = 0; j < 131; j++) {
            for (int k = 0; k < 91; k++) {
                f[i][j][k][0] = -inf;
                f[i][j][k][1] = inf;
            }
        }
    }
    printf("%d\n", dp(0, M, 1, 0));

    return 0;
}
```

{% endcollapse %}

<br>

# [[ZJOI2012]波浪](https://www.luogu.com.cn/problem/P2612)

咕

{% collapse Tag %}

{% endcollapse %}

{% collapse Solution %}

{% endcollapse %}

{% collapse Code %}

{% endcollapse %}

<br>

# [Game on Chessboard (CCPC 2019 秦皇岛 G)](http://codeforces.com/gym/102361/problem/G)

咕

{% collapse Tag %}

{% endcollapse %}

{% collapse Solution %}

{% endcollapse %}

{% collapse Code %}

{% endcollapse %}

<br>

# [Fountains (ICPC 2020 上海 F)](https://ac.nowcoder.com/acm/contest/9925/F)

咕

{% collapse Tag %}

{% endcollapse %}

{% collapse Solution %}

{% endcollapse %}

{% collapse Code %}

{% endcollapse %}

<br>

# [Sum (VK Cup 2019 Final D)](http://codeforces.com/contest/1442/problem/D)

{% collapse Tag %}

背包，分治

{% endcollapse %}

{% collapse Solution %}

首先来看题目给出的特殊性质，每个数组单独来讲都是非递减的。

因此有一个结论，在最优方案中，不存在两个数组同时未取完的情况。使用反证法，如果存在两个数组同时未取完，这两个数组最后取的数分别为$x$和$y$，则若$x>y$，可以把$y$换成$x$右边的数，$x<y$则可以把$x$换成$y$右边的数。

换句话说，最优方案可以视作取完若干个数组，再外加一个取到一半的数组。

那么就诞生了一个暴力做法，枚举那个取到一半的数组，其余$N-1$个数组相当于$N-1$个物品，做一下背包，然后再枚举一下多余的一个数组里取了几个数，和背包结合一下出答案。

可是我们没有办法高效地从一个$N$个物品的背包得到删去里面某一个物品后剩余物品组成的背包。如果暴力做$N$次背包的话。$O(N\times N\times K)$的时间复杂度是不合格的。

然后就是这道题的神奇分治优化。

优化思路是这样的，我们将去除物品$i$的影响后剩下的背包称作$f[i]$，可以发现，$f[\frac{N}{2}+1]...f[N]$有个共同点，那就是都保留了物品$i\in [1, \frac{N}{2}]$对背包的影响，所以可以先求出物品$i\in [1, \frac{N}{2}]$组成的背包，然后再由该背包去推出$f[\frac{N}{2}+1]...f[N]$。

将上面的优化思路推广，想象成一颗线段树的形式，线段树上的某一个结点$x[l,r]$表示的就是去除物品$i\in [l,r]$的影响后的背包。递归下去时，往左儿子的背包中加入右儿子区间内的所有物品的影响，往右儿子的背包中加入左儿子区间内的所有物品的影响。递归到最后$l==r$的叶子结点时，就是我们需要的$f[i]$。

空间上，因为是类似线段树的形式，总共$4\times N$个结点，每个节点一个大小为$K$的背包，空间$O(4\times N\times K)$。

时间上，线段树总共$logN$层，在每一层中，每个物品都会跟其兄弟结点中的背包发生一次合并，因此每一层都是$N$次合并，每次合并$O(K)$，因此时间复杂度$O(logN\times N\times K)$。

{% endcollapse %}

{% collapse Code %}

```c++
#include <bits/stdc++.h>
using namespace std;

typedef pair<int, int> pi;
typedef long long ll;

const int maxn = 3005;
const int inf = 0x3f3f3f3f;
const int mod = 1e9 + 7;

vector<int> a[maxn];
int t[maxn];
ll sum[maxn];

int n, k;

struct Info {
    ll f[maxn];
    void copy(Info tmp) {
        for (int i = 0; i <= k; i++) {
            f[i] = tmp.f[i];
        }
    }
    void dp(ll v, int c) {
        for (int i = k; i >= c; i--) {
            f[i] = max(f[i], f[i - c] + v);
        }
    }
} info[maxn << 2];

ll pre[1000005];
ll ans;

ll calc(vector<int> &v, int vn, ll f[]) {
    ll ret = 0;
    for (int i = 1; i <= vn; i++) {
        pre[i] = pre[i - 1] + v[i];
    }
    for (int i = 0; i <= min(vn, k); i++) {
        ret = max(ret, f[k - i] + pre[i]);
    }
    return ret;
}

void solve(int x, int l, int r) {
    if (l == r) {
        ans = max(ans, calc(a[l], t[l], info[x].f));
    } else {
        int mid = l + r >> 1;
        int ls = x * 2, rs = ls + 1;
        info[ls].copy(info[x]);
        info[rs].copy(info[x]);
        for (int i = mid + 1; i <= r; i++) {
            info[ls].dp(sum[i], t[i]);
        }
        for (int i = l; i <= mid; i++) {
            info[rs].dp(sum[i], t[i]);
        }
        solve(ls, l, mid);
        solve(rs, mid + 1, r);
    }
}

void run() {
    scanf("%d%d", &n, &k);
    for (int i = 1; i <= n; i++) {
        scanf("%d", t + i);
        a[i].push_back(0);
        for (int j = 1; j <= t[i]; j++) {
            int x; scanf("%d", &x);
            a[i].push_back(x);
            sum[i] += x;
        }
    }
    solve(1, 1, n);
    printf("%lld\n", ans);
}

int main() {

    int T = 1;
//    scanf("%d", &T);
    while (T--) {
        run();
    }

    return 0;
}
```

{% endcollapse %}

<br>

# [Lucky Numbers (CF 1428 G)](http://codeforces.com/contest/1428/problem/G)

咕

{% collapse Tag %}

{% endcollapse %}

{% collapse Solution %}

{% endcollapse %}

{% collapse Code %}

{% endcollapse %}

<br>

# [String Transformation (CF 1383 C)](http://codeforces.com/contest/1383/problem/C)

咕

{% collapse Tag %}

{% endcollapse %}

{% collapse Solution %}

{% endcollapse %}

{% collapse Code %}

{% endcollapse %}

<br>

# [Cells Blocking (300iq contest)](http://codeforces.com/gym/102538/problem/C)

咕

{% collapse Tag %}

{% endcollapse %}

{% collapse Solution %}

{% endcollapse %}

{% collapse Code %}

{% endcollapse %}