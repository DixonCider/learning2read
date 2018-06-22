#include "csv.h"
#include <valarray>
#include <iostream>
#include <random>
#include <unordered_map>
#include <cstring>
#include <cmath>

struct Data {
  int user, book;
  double rating, weight;
};

std::vector<Data> Input(const std::string& filename) {
  auto raw = ReadCsv(filename);
  std::vector<Data> ret;
  int usr = -1, bok = -1, rat = -1, trat = -1;
  for (size_t i = 0; i < raw[0].size(); i++) {
    if (raw[0][i] == "User-ID") usr = i;
    else if (raw[0][i] == "ISBN") bok = i;
    else if (raw[0][i] == "Weight") trat = i;
    else rat = i;
  }
  for (size_t i = 1; i < raw.size(); i++) {
    ret.push_back({std::stoi(raw[i][usr]), std::stoi(raw[i][bok]),
        ~rat ? std::stod(raw[i][rat]) : -1., std::stod(raw[i][trat])});
  }
  return ret;
}

typedef std::vector<std::valarray<double>> Matrix;

inline double Pred(const Data& d, const Matrix& W, const Matrix& V) {
  return (W[d.user] * V[d.book]).sum();
}

double Err(const std::vector<Data>& dat, const Matrix& W, const Matrix& V,
           bool track2, bool trunc = false) {
  double toterr = 0;
  size_t count = 0;
  for (auto& i : dat) {
    double pred = Pred(i, W, V);
    if (trunc) pred = (int)(pred + 0.5);
    if (track2) toterr += fabs(pred - i.rating) / i.weight * 100;
    else toterr += fabs(pred - i.rating);
    count++;
  }
  return toterr / count;
}

inline void Update(const Data& d, Matrix& W, Matrix& V, double step, bool track2, int dimen) {
  double pred = Pred(d, W, V);
  if (track2) step /= d.weight;
  if (pred > d.rating) step = -step;
  W[d.user] += step * V[d.book];
  V[d.book] += step * W[d.user];
  double lim = 4. / sqrt(dimen);
  for (auto& i : W[d.user]) if (i > lim) i = lim; else if (i < -lim) i = -lim;
  for (auto& i : V[d.book]) if (i > lim) i = lim; else if (i < -lim) i = -lim;
}


const int kUM = 110003, kBM = 350003;
bool usrmp[kUM], bokmp[kBM];

template <class T>
void Train(std::vector<Data> dat, const std::vector<Data>& rawtest, int dimen,
           int iter, double step, bool track2, T& gen) {
  std::vector<Data> test;
  memset(usrmp, 0, kUM);
  memset(bokmp, 0, kBM);
  for (auto& i : dat) {
    usrmp[i.user] = true;
    bokmp[i.book] = true;
  }
  for (auto& i : rawtest)
    if (usrmp[i.user] && bokmp[i.book])
      test.push_back({i.user, i.book, i.rating, i.weight});
  std::cout << test.size() << ' ' << rawtest.size() << '\n';

  Matrix W(kUM, std::valarray<double>(dimen));
  Matrix V(kBM, std::valarray<double>(dimen));
  std::cout << Err(dat, W, V, track2) << ' ' << Err(test, W, V, track2) << std::endl;

  std::uniform_real_distribution<double> rrand(-1. / sqrt(dimen), 1. / sqrt(dimen));
  {
    for (auto& i : W)
      for (auto& j : i) j = rrand(gen);
    for (auto& i : V)
      for (auto& j : i) j = rrand(gen);
  }
  std::cout << Err(dat, W, V, track2) << ' ' << Err(test, W, V, track2) << std::endl;

  int Q = iter / 20;
  std::uniform_int_distribution<size_t> mrand(0, dat.size() - 1);
  for (int i = 0; i < iter; i++) {
    int id = mrand(gen);
    if (track2) {
      while (fabs(rrand(gen)) > 1. / dat[id].rating) id = mrand(gen);
    }
    Update(dat[id], W, V, step, track2, dimen);
    if (i % Q == Q - 1) {
      std::cout << Err(dat, W, V, track2) << ' ' << Err(test, W, V, track2) << ';';
      std::cout << Err(dat, W, V, track2, true) << ' ' << Err(test, W, V, track2, true) << '\n';
    }
  }
}

int main(int argc, char** argv) {
  if (argc < 7) {
    std::cout << "./MF track(1,2) suf dimen iter step testnum [seed]\n";
    return 1;
  }
  std::mt19937_64 gen(argc >= 8 ? std::stoull(argv[7]) : 0);
  bool track2 = std::string(argv[1]) == "2";
  std::string suf(argv[2]);
  int kDimen = std::stoi(std::string(argv[3]));
  int kIter = std::stoi(std::string(argv[4]));
  double kStep = std::stod(std::string(argv[5]));
  int kTest = std::stoi(std::string(argv[6]));

  std::vector<Data> train, test;
  {
    auto all = Input("train-" + suf + ".csv");
    std::shuffle(all.begin(), all.end(), gen);
    train.assign(all.begin(), all.end() - kTest);
    test.assign(all.end() - kTest, all.end());
  }

  Train(train, test, kDimen, kIter, kStep, track2, gen);
}
