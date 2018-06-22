#ifndef SIMULATED_ANNEALING_H_
#define SIMULATED_ANNEALING_H_

#include <cmath>
#include <random>
#include <functional>

template <class T, class Gen = std::mt19937_64, class ResType = double> class SimulatedAnnealing {
private:
  typedef std::function<ResType(const T&)> EvalFuncType_;
  typedef std::function<T(const T&)> NeighborFuncType_;
  std::uniform_real_distribution<double> rand_ = std::uniform_real_distribution<double>(0., 1.);
  T now_, best_;
  double nowres_, bestres_;
  EvalFuncType_ eval_;
  NeighborFuncType_ next_;
  Gen gen_;
public:
  SimulatedAnnealing(EvalFuncType_ eval, NeighborFuncType_ nxt, const T& start = T(), Gen g = Gen())
      : now_(start), best_(start), eval_(eval), next_(nxt), gen_(g) {
    nowres_ = bestres_ = eval_(now_);
  }

  double DoIter(double temp) {
    T next = next_(now_);
    double res = eval_(next);
    if (res < nowres_ || std::exp((nowres_ - res) / temp) > rand_(gen_)) {
      now_ = next;
      nowres_ = res;
      if (res < bestres_) best_ = now_, bestres_ = nowres_;
    }
    return nowres_;
  }

  const T& GetNow() const { return now_; }
  const T& GetBest() const { return best_; }
  double GetNowVal() const { return nowres_; }
  double GetBestVal() const { return bestres_; }

  void Anneal(double start, double end, int iter) {
    for (int i = 0; i < iter; i++)
      DoIter((start * (iter - i) + end * i) / iter);
  }
};

#endif
