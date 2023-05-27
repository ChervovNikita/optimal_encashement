#include <iostream>
#include <vector>

using namespace std;

const int max_wait = 14;
const int max_cash = 1000000;
const double INF = 1e9;

inline double loss(double was, double add, bool take) {
    if (take) {
        return max(100., was / 1e4) + add * 2 / 100 / 365;
    } else {
        if (was >= max_cash) return INF;
        return (was + add) * 2 / 100 / 365;
    }
}

signed main() {
    int days, wait;
    cin >> days >> wait;
    int begin_with; cin >> begin_with;
    vector<int> add(days - 1);
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
        double cur_cash = begin_with;

        for (int day = 0; day < days; ++day) {
            ++current_wait;
            cur_loss += loss(cur_cash, add[day], mask & (1 << day));
            if (mask & (1 << day)) {
                first_take = min(first_take, day);
                current_wait = 0;
                cur_cash = 0;
            }
            cur_cash += add[day];
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
//        cout << begin_with << ' ';
//        for (int i = 0; i < days; ++i) {
//            cout << (bool)(best_skip_mask & (1 << i)) << ' ';
//            if (best_skip_mask & (1 << i)) begin_with = 0;
//            begin_with += add[i];
//            cout << begin_with << ' ';
//        }
//        cout << '\n' << best_skip_loss << '\n';
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
