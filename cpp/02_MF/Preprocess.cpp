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

void Init(const std::vector<Data>& dat, double defu, double defb) {
  for (int i = 0; i < kUM; i++) usrsum[i] = usrcnt[i] = 0;
  for (int i = 0; i < kBM; i++) boksum[i] = bokcnt[i] = 0;

  auto upd = [](auto a, int z, const Data& i) { a.first[z] += i.rating; a.second[z]++; };
  const std::pair<double*, int*> U(usrsum, usrcnt), B(boksum, bokcnt);

  for (auto& i : dat) {
    upd(U, i.user, i);
    upd(B, i.book, i);
  }
  for (int i = 0; i < kUM; i++) usrsum[i] = usrcnt[i] ? usrsum[i] / usrcnt[i] : defu;
  for (int i = 0; i < kBM; i++) boksum[i] = bokcnt[i] ? boksum[i] / bokcnt[i] : defb;
}

int main(int argc, char** argv) {
  if (argc < 8) {
    std::cout << "./MF track2 suf usrb bokb bias defu defb\n";
    return 1;
  }
  bool track2 = std::string(argv[1]) == "2";
  std::string suf(argv[2]);
  double usrb = std::stod(std::string(argv[3]));
  double bokb = std::stod(std::string(argv[4]));
  double bias = std::stod(std::string(argv[5]));
  double defu = std::stod(std::string(argv[6]));
  double defb = std::stod(std::string(argv[7]));

  auto train = Input("data/clean/number_train.csv");
  auto test = Input("data/clean/number_test.csv");
  std::fstream ftrain("train-" + suf + ".csv", std::ios::out);
  std::fstream ftest("test-add-" + suf + ".csv", std::ios::out);

  Init(train, defu, defb);
  for (auto& i : test) {
    double pred = usrsum[i.user] * usrb + boksum[i.book] * bokb + bias;
    if (pred > 10) pred = 10;
    if (pred < 1) pred = 1;
    if (track2) ftest << pred << '\n';
    else ftest << (int)(pred + 0.5) << '\n';
  }
  ftrain << "User-ID,ISBN,Weight,RatingResidue\n";
  for (auto& i : train) {
    double pred = usrsum[i.user] * usrb + boksum[i.book] * bokb + bias;
    if (pred > 10) pred = 10;
    if (pred < 1) pred = 1;
    ftrain << i.user << ',' << i.book << ',' << i.rating << ',';
    if (track2) ftrain << i.rating - pred << '\n';
    else ftrain << i.rating - (int)(pred + 0.5) << '\n';
  }
}
