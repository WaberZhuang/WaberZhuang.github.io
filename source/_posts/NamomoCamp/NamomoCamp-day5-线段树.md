---
title: NamomoCamp-day5-线段树
date: 2021-02-01 07:28:16
categories: [NamomoCamp]
tags:
mathjax: true
---

Solution for NamomoCamp2021-div2-day5-线段树

<!-- more -->

# [Subsequence Count (hdu 6155)](http://acm.hdu.edu.cn/showproblem.php?pid=6155)

{% collapse Tag %}

dp， 矩阵，线段树

{% endcollapse %}

{% collapse Solution %}

```
f[i][0] 考虑前 i 位，以 0 结尾的本质不同子序列个数
f[i][1] 考虑前 i 位，以 1 结尾的本质不同子序列个数
s[i]==0 -> f[i][0]=f[i-1][0]+f[i-1][1]+1;
           f[i][1]=f[i-1][1];
s[i]==1 -> f[i][1]=f[i-1][0]+f[i-1][1]+1;
           f[i][0]=f[i-1][0];
转化为矩阵乘法
  mat   -> [f[i][0] f[i][1] 1]
  初值   -> [0 0 1]
  
s[i]==0 -> [1 0 0]
           [1 1 0]
           [1 0 1]

s[i]==1 -> [1 1 0]
           [0 1 0]
           [0 1 1]
info = 转移矩阵乘积
tag = flip, flip 区间 = 每个单点的矩阵 swap 1,2行和1,2列 = 矩阵乘积swap 1,2行和1,2列 (横竖编号均变化 [1,2,3]->[2,1,3] )
info合并: 矩阵乘
tag合并: 异或
tag更新info: 矩阵交换
query得出矩阵乘mul, [0 0 1] * mul = [f[0] f[1] 1]
ans = f[0] + f[1]
```

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

struct mat {
#define N 3
    ll w[N][N];

    mat() {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                w[i][j] = 0;
            }
        }
    }
    mat(int _w[][N]) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                w[i][j] = _w[i][j];
            }
        }
    }

    mat operator * (mat m) {
        mat ret;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    ret.w[i][j] += w[i][k] * m.w[k][j] % mod;
                    ret.w[i][j] %= mod;
                }
            }
        }
        return ret;
    }

    void swapRow(int x, int y) {
        for (int j = 0; j < N; j++) {
            swap(w[x][j], w[y][j]);
        }
    }
    void swapColumn(int x, int y) {
        for (int i = 0; i < N; i++) {
            swap(w[i][x], w[i][y]);
        }
    }
#undef N
};

int w[2][3][3] = {
        {
                {1, 0, 0},
                {1, 1, 0},
                {1, 0, 1}
        },
        {
                {1, 1, 0},
                {0, 1, 0},
                {0, 1, 1}
        }
};
int c[3][3] = {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}
};
char s[maxn];

struct Tag {
    int rev;
    Tag() {}
    Tag(int _rev) {
        rev = _rev;
    }
    void clear() {
        rev = 0;
    }
    void mergeTag(Tag k) {
        rev ^= k.rev;
    }
} tag[maxn << 2];

struct Info {
    mat m;
    Info() {
        m = mat(c);
    }
    Info(mat _m) {
        m = _m;
    }
    void applyTag(Tag k) {
        if (k.rev) {
            m.swapRow(0, 1);
            m.swapColumn(0, 1);
        }
    }
} info[maxn << 2];

Info mergeInfo(Info x, Info y) {
    return Info(x.m * y.m);
}

void applyTag(int x, Tag k) {
    tag[x].mergeTag(k);
    info[x].applyTag(k);
}

void pushup(int x) {
    info[x] = mergeInfo(info[x * 2], info[x * 2 + 1]);
}

void pushdown(int x) {
    applyTag(x * 2, tag[x]);
    applyTag(x * 2 + 1, tag[x]);
    tag[x].clear();
}

void build(int x, int l, int r) {
    if (l == r) {
        info[x].m = mat(w[s[l] - '0']);
    } else {
        int mid = l + r >> 1;
        build(x * 2, l, mid);
        build(x * 2 + 1, mid + 1, r);
        pushup(x);
    }
    tag[x].rev = 0;
}

void update(int x, int l, int r, int ql, int qr, Tag k) {
    if (ql > r || qr < l) return;
    if (ql <= l && r <= qr) {
        applyTag(x, k);
        return;
    }
    pushdown(x);
    int mid = l + r >> 1;
    update(x * 2, l, mid, ql, qr, k);
    update(x * 2 + 1, mid + 1, r, ql, qr, k);
    pushup(x);
}

Info query(int x, int l, int r, int ql, int qr) {
    if (ql > r || qr < l) return Info();
    if (ql <= l && r <= qr) {
        return info[x];
    }
    pushdown(x);
    int mid = l + r >> 1;
    return mergeInfo(query(x * 2, l, mid, ql, qr), query(x * 2 + 1, mid + 1, r, ql, qr));
}

int main() {

    int T;
    scanf("%d", &T);
    while (T--) {
        int n, q;
        scanf("%d%d", &n, &q);
        scanf("%s", s + 1);
        build(1, 1, n);
        while (q--) {
            int op, l, r;
            scanf("%d%d%d", &op, &l, &r);
            if (op == 1) {
                update(1, 1, n, l, r, Tag(1));
            } else {
                mat m;
                m.w[0][2] = 1;
                m = m * query(1, 1, n, l, r).m;
                printf("%lld\n", (m.w[0][0] + m.w[0][1]) % mod);
            }
        }
    }

    return 0;
}
```

{% endcollapse %}