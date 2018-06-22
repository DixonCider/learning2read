#include "csv.h"
#include "simulated_annealing.h"
#include <valarray>
#include <iostream>
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
const int kFold = 24;
double usrsum[kFold][kUM], boksum[kFold][kBM];
int usrcnt[kFold][kUM], bokcnt[kFold][kBM];

void Init(const std::vector<Data>& dat, int f) {
  auto upd = [](auto a, int z, const Data& i) { a.first[z] += i.rating; a.second[z]++; };
  const std::pair<double*, int*> U(usrsum[f], usrcnt[f]), B(boksum[f], bokcnt[f]);

  for (auto& i : dat) {
    upd(U, i.user, i);
    upd(B, i.book, i);
  }
  for (int i = 0; i < kUM; i++) usrsum[f][i] = usrcnt[f][i] ? usrsum[f][i] / usrcnt[f][i] : 0;
  for (int i = 0; i < kBM; i++) boksum[f][i] = bokcnt[f][i] ? boksum[f][i] / bokcnt[f][i] : 0;
}

void Change(double defu, double defb) {
#pragma omp parallel for
  for (int k = 0; k < kFold; k++)
    for (int i = 0; i < kUM; i++)
      if (!usrcnt[k][i]) usrsum[k][i] = defu;
#pragma omp parallel for
  for (int k = 0; k < kFold; k++)
    for (int i = 0; i < kBM; i++)
      if (!bokcnt[k][i]) boksum[k][i] = defb;
}

double Check(const std::vector<Data>& test, int t, double usrb, double bokb,
             double bias, bool track2) {
  double err = 0;
  for (auto& i : test) {
    double tmp = fabs(bias + usrsum[t][i.user] * usrb + boksum[t][i.book] * bokb - i.rating);
    err += track2 ? tmp / i.rating : tmp;
  }
  if (track2) err *= 5;
  return err / test.size();
}

std::vector<Data> train[kFold], test[kFold];

struct State {
  double usrb, bokb, bias, defu, defb;
  bool track2;
};

double Calculate(const State& t) {
  Change(t.defu, t.defb);

  double toterr = 0;
#pragma omp parallel for reduction(+:toterr)
  for (int i = 0; i < kFold; i++) {
    toterr += Check(test[i], i, t.usrb, t.bokb, t.bias, t.track2);
  }
  return toterr / kFold;
}

std::mt19937_64 gen;

State Neighbor(State f) {
  typedef std::uniform_real_distribution<double> mrand;
  f.usrb += mrand(-0.01, 0.01)(gen);
  f.bokb += mrand(-0.01, 0.01)(gen);
  f.bias += mrand(-0.1, 0.1)(gen);
  f.defu += mrand(-0.1, 0.1)(gen);
  f.defb += mrand(-0.1, 0.1)(gen);
  auto chk = [](double& a, double l, double r) {
    if (a < l) a = l;
    else if (a > r) a = r;
  };
  chk(f.usrb, -0.2, 1.2);
  chk(f.bokb, -0.2, 1.2);
  chk(f.bias, -2, 2);
  chk(f.defu, 4, 10);
  chk(f.defb, 4, 10);
  return f;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cout << "./MF track(1,2) stemp iter [seed]\n";
    return 1;
  }
  gen.seed(argc >= 5 ? std::stoull(argv[4]) : 0);
  bool track2 = std::string(argv[1]) == "2";
  double start = std::stod(std::string(argv[2]));
  int iter = std::stoi(std::string(argv[3]));

  auto all = Input("data/clean/number_train.csv");
  std::shuffle(all.begin(), all.end(), gen);

  for (int t = 0; t < kFold; t++) {
    int L = all.size() * t / kFold, R = all.size() * (t + 1) / kFold, N = all.size();
    for (int i = 0; i < L; i++) train[t].push_back(all[i]);
    for (int i = L; i < R; i++) test[t].push_back(all[i]);
    for (int i = R; i < N; i++) train[t].push_back(all[i]);
    Init(train[t], t);
  }

  State st = {0., 0., 0., 6., 6., track2};
  SimulatedAnnealing<State> sa(Calculate, Neighbor, st, gen);

  for (int i = 0; i < 40; i++) {
    double L = start * (40 - i) / 40;
    double R = start * (39 - i) / 40;
    sa.Anneal(L, R, iter / 40);
    std::cout << R << ' ' << sa.GetNowVal() << ' ' << sa.GetBestVal() << std::endl;
  }
  auto f = sa.GetBest();
  std::cout << f.usrb << ' ' << f.bokb << ' ' << f.bias << ' '
      << f.defu << ' ' << f.defb << '\n';
}
