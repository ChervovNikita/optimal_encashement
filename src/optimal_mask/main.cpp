#include <iostream>
#include <vector>

using namespace std;

const int max_wait = 14;
const int max_cash = 1000000;
const double INF = 1e9;

inline double loss(double was, bool take) {
    if (take) {
        return max(100., was / 1e4);
    } else {
        if (was >= max_cash) return INF;
        return was * 2 / 100 / 365;
    }
}

int sum(vector<int>& p, int l, int r) {
    if (!l) return p[r];
    return p[r] - p[l - 1];
}

double skip(vector<int>& p, int l, int r, int extra = 0) {
    double res = 0;
    int current = extra;
    for (int i = l; i < r; ++i) {
        res += loss(current, 0);
        current += p[i];
    }
    return res;
}

signed main() {
    int days, wait;
    cin >> days >> wait;
//    double alpha, beta;
//    cin >> alpha >> beta;

    int begin_with; cin >> begin_with;
    vector<int> add(days);
    for (int i = 0; i < days; ++i) {
        cin >> add[i];
    }

    vector<int> pref = add;
    for (int i = 1; i < days; ++i) {
        pref[i] += pref[i - 1];
    }

    vector<vector<double>> dp(days, vector<double>(2, INF));

    dp[0][0] = loss(begin_with, 0);
    dp[0][1] = loss(begin_with, 1);

    for (int i = 1; i < days; ++i) {
        if (i <= wait) {
            dp[i][1] = min(dp[i][1], skip(add, 0, i, begin_with) + loss(begin_with + sum(pref, 0, i - 1), 1));
        }
        for (int last = max(0, i - max_wait); last < i; ++last) {
            dp[i][0] = min(dp[i][0], dp[last][1] + skip(add, last, i));
            dp[i][1] = min(dp[i][1], dp[last][1] + skip(add, last, i - 1) + loss(sum(pref, last, i - 1), 1));
        }
    }

    double full_res = max(dp[days - 1][0], dp[days - 1][1]);

    dp = vector<vector<double>>(days, vector<double>(2, INF));

    dp[0][0] = loss(begin_with, 0);

    for (int i = 1; i < days; ++i) {
        if (i <= wait) {
            dp[i][1] = min(dp[i][1], skip(add, 0, i, begin_with) + loss(begin_with + sum(pref, 0, i - 1), 1));
        }
        for (int last = max(0, i - max_wait); last < i; ++last) {
            dp[i][0] = min(dp[i][0], dp[last][1] + skip(add, last, i));
            dp[i][1] = min(dp[i][1], dp[last][1] + skip(add, last, i - 1) + loss(sum(pref, last, i - 1), 1));
        }
    }
    double res = max(dp[days - 1][0], dp[days - 1][1]);

    cout << res << '\n';
    if (abs(full_res - res) < 1e-6) {
        cout << "0 0\n";
    } else {
        cout << "1 " << res - full_res << '\n';
    }
}
