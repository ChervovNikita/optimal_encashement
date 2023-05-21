#include <iostream>
#include <fstream>
#include <string>

using namespace std;

const int n = 1630;

double a[n][n], w[n][n];

int main() {
    freopen("data/processed/data.txt", "r", stdin);
//    freopen("result.txt", "w", stdout);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cin >> a[i][j];
            w[i][j] = a[i][j];
        }
    }

    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                w[i][j] = min(w[i][j], w[i][k] + w[k][j]);
            }
        }
    }

    double sum = 0;
    double cnt = 0;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            sum += abs(w[i][j] - a[i][j]);
            cnt += 1;
        }
//        cout << '\n';
    }
    cout << sum / cnt << '\n';
}
