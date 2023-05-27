// #include <bits/stdc++.h>
#include "vector"
#include "iostream"
#include "algorithm"

using namespace std;

const int max_wait = 14;
const int max_cash = 1000000;
const double INF = 1e9;

inline double loss(double x, bool take) {
    if (take) {
        return max(100., x / 1e4);
    } else {
        if (x >= max_cash) return INF;
        return x * 2 / 100 / 365;
    }
}

signed main() {
    int days, wait;
    cin >> days >> wait;
    vector<int> add(days);
    for (int i = 0; i < days; ++i) {
        cin >> add[i];
    }
    double best_loss = INF;
    int best_mask = 0;
    double best_skip_loss = INF;
    int best_skip_mask = 0;

    for (int mask = 0; mask < (1 << days); ++mask) {
        int first_take = max_wait + 10;
        int current_wait = 0;
        double cur_loss = 0;
        double cur_cash = 0;

        for (int day = 0; day < days; ++day) {
            ++current_wait;
            cur_cash += add[day];
            cur_loss += loss(cur_cash, mask & (1 << day));
            if (mask & (1 << day)) {
                first_take = min(first_take, day);
                current_wait = 0;
                cur_cash = 0;
            }
            if (current_wait == max_wait) {
                cur_loss += INF;
            }
        }
        if (first_take > wait) {
            cur_loss += INF;
        }
        if (mask & 1) {
            if (best_loss > cur_loss) {
                best_loss = cur_loss;
                best_mask = mask;
            }
        } else {
            if (best_skip_loss > cur_loss) {
                best_skip_loss = cur_loss;
                best_skip_mask = mask;
            }
        }
    }


    if (best_skip_loss < best_loss) {
        cout << "0 0\n";
//        for (int i = 0; i < days; ++i) {
//            cout << (bool)(best_skip_mask & (1 << i));
//        }
    } else {
        cout << "1 " << best_skip_loss - best_loss << '\n';
//        for (int i = 0; i < days; ++i) {
//            cout << (bool)(best_mask & (1 << i));
//        }
//        cout << '\n';
//        for (int i = 0; i < days; ++i) {
//            cout << (bool)(best_skip_mask & (1 << i));
//        }
    }
}