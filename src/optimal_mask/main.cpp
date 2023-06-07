#include <iostream>
#include <vector>

using namespace std;

const int max_wait = 14;
const int max_cash = 1000000;
const double INF = 1e9;
double interest = 2. / 100 / 365;

double loss(double was, bool take) {
    if (take) {
        return max(100., was / 1e4);
    } else {
        if (was > max_cash) {
            return INF;
        }
        return was * interest;
    }
}

double skip(vector<int>& a, int l, int r) {
    double current = 0;
    double res = 0;
    for (int i = l; i <= r; ++i) {
        current += a[i];
        res += loss(current, false);
    }
    return res;
}

double sum(vector<int>& p, int l, int r) {
    if (!l) return p[r];
    return p[r] - p[l - 1];
}

signed main() {
    int days, wait;
    cin >> days >> wait;
    vector<int> adds(days);
    for (auto& x : adds) {
        cin >> x;
    }

    vector<int> pref = adds;
    for (int i = 1; i < pref.size(); ++i) {
        pref[i] += pref[i - 1];
    }

    vector<vector<double> > dp(days, vector<double>(2, INF));
    dp[0][1] = loss(adds[0], true);
    for (int i = 1; i < days; ++i) {
//        if (i <= max_wait) {
//            dp[i][0] = min(dp[i][0], skip(adds, 0, i));
//            dp[i][1] = min(dp[i][1], skip(adds, 0, i - 1) + loss(sum(pref, 0, i), true));
//        }
        for (int last = max(0, i - max_wait - 1); last < i; ++last) {
            dp[i][0] = min(dp[i][0], dp[last][1] + skip(adds, last + 1, i));
            dp[i][1] = min(dp[i][1], dp[last][1] + skip(adds, last + 1, i - 1) + loss(sum(pref, last + 1, i), true));
        }
    }
    double take_res = min(dp[days - 1][0], dp[days - 1][1]);

    dp = vector<vector<double> >(days, vector<double>(2, INF));
    dp[0][0] = loss(adds[0], false);
    for (int i = 1; i < days; ++i) {
        if (i <= max_wait) {
            dp[i][0] = min(dp[i][0], skip(adds, 0, i));
            dp[i][1] = min(dp[i][1], skip(adds, 0, i - 1) + loss(sum(pref, 0, i), true));
        }
        for (int last = max(0, i - max_wait - 1); last < i; ++last) {
            dp[i][0] = min(dp[i][0], dp[last][1] + skip(adds, last + 1, i));
            dp[i][1] = min(dp[i][1], dp[last][1] + skip(adds, last + 1, i - 1) + loss(sum(pref, last + 1, i), true));
        }
    }
    double skip_res = min(dp[days - 1][0], dp[days - 1][1]);

    if (skip_res > INF / 2 || take_res > INF / 2) {
        cout << "0\n";
    } else {
        cout << take_res - skip_res << '\n';
    }
}