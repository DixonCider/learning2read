#include "csv.h"
#include <valarray>
#include <iostream>
#include <random>
#include <unordered_map>
#include <omp.h>

struct Data {
  int user, book;
  double rating;
};

std::vector<Data> Input(const std::string& filename) {
  auto raw = ReadCsv(filename);
  std::vector<Data> ret;
  int usr = -1, bok = -1, rat = -1;
  for (size_t i = 0; i < raw[0].size(); i++) {
    if (raw[0][i] == "User-ID") usr = i;
    else if (raw[0][i] == "ISBN") bok = i;
    else rat = i;
  }
  for (size_t i = 1; i < raw.size(); i++) {
    ret.push_back({std::stoi(raw[i][usr]), std::stoi(raw[i][bok]),
        ~rat ? std::stod(raw[i][rat]) : -1.});
  }
  return ret;
}

const int kUM = 110003, kBM = 350003;
double usrsum[kUM], boksum[kBM];
int usrcnt[kUM], bokcnt[kBM];
double tot;

void Init(const std::vector<Data>& dat, double defu, double defb) {
  for (int i = 0; i < kUM; i++) usrsum[i] = usrcnt[i] = 0;
  for (int i = 0; i < kBM; i++) boksum[i] = bokcnt[i] = 0;

  auto upd = [](auto a, int z, const Data& i) { a.first[z] += i.rating; a.second[z]++; };
  const std::pair<double*, int*> U(usrsum, usrcnt), B(boksum, bokcnt);

  tot = 0;
  for (auto& i : dat) {
    upd(U, i.user, i);
    upd(B, i.book, i);
    tot += i.rating;
  }
  tot /= dat.size();
  for (int i = 0; i < kUM; i++) usrsum[i] = usrcnt[i] ? usrsum[i] / usrcnt[i] : defu;
  for (int i = 0; i < kBM; i++) boksum[i] = bokcnt[i] ? boksum[i] / bokcnt[i] : defb;
}

double Check(const std::vector<Data>& test, double usrb, double bokb,
             double totalb, bool track2) {
  double err = 0;
  for (auto& i : test) {
    double tmp = fabs(tot * totalb + usrsum[i.user] * usrb + boksum[i.book] * bokb - i.rating);
    err += track2 ? tmp / i.rating : tmp;
  }
  if (track2) err *= 100;
  return err / test.size();
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cout << "./MF track(1,2) fold [seed]\n";
    return 1;
  }
  std::mt19937_64 gen(argc >= 4 ? std::stoull(argv[3]) : 0);
  bool track2 = std::string(argv[1]) == "2";
  int kFold = std::stoi(std::string(argv[2]));

  std::vector<Data> train, test;
  auto all = Input("data/clean/number_train.csv");
  std::shuffle(all.begin(), all.end(), gen);

  //const double aS = 6.5, dS = 5.5, cS = 0.85, bS = 0.2; // track2
  //const double aP = 0.1, dP = 0.1, cP = 0.01, bP = 0.01;
  //const int aN = 10, dN = 10, cN = 10, bN = 10;
  const double aS = 7, dS = 6.4, cS = 0.85, bS = 0.11; // track1
  const double aP = 0.2, dP = 0.2, cP = 0.02, bP = 0.02;
  const int aN = 10, dN = 10, cN = 10, bN = 10;
  double param[aN+1][dN+1][bN+1][cN+1]; // defu, defb, bookb, userb
  for (auto& i : param)
    for (auto& j : i)
      for (auto& k : j)
        for (auto& l : k) l = 0;
  for (int t = 0; t < kFold; t++) {
    train.clear(); test.clear();
    int L = all.size() * t / kFold, R = all.size() * (t + 1) / kFold, N = all.size();
    for (int i = 0; i < L; i++) train.push_back(all[i]);
    for (int i = L; i < R; i++) test.push_back(all[i]);
    for (int i = R; i < N; i++) train.push_back(all[i]);

    for (int a = 0; a <= aN; a++) {
      for (int d = 0; d <= dN; d++) {
        Init(train, a * aP + aS, d * dP + dS);
        std::cout << tot << std::endl;
        #pragma omp parallel for collapse(2)
        for (int b = 0; b <= bN; b++) {
          for (int c = 0; c <= cN; c++) {
            double k = c * cP + cS, j = b * bP + bS;
            param[a][d][b][c] += Check(test, k, j, 1 - k - j, track2);
          }
        }
      }
      std::cerr << t << ' ' << a << '\n';
    }
  }
  int a, b, c, d;
  double i, j, k, l;
  for (a = 0, i = aS; a <= 10; a++, i += aP)
    for (d = 0, l = dS; d <= 10; d++, l += dP)
      for (b = 0, j = bS; b <= 10; b++, j += bP)
        for (c = 0, k = cS; c <= 10; c++, k += cP)
          std::cout << param[a][d][b][c] / kFold << ' ' << i << ' ' << l << ' ' << j << ' ' << k << '\n';
}
