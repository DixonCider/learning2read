#include <vector>
#include <string>
#include <fstream>
#include <algorithm>

// split a comma-separated line into columns
std::vector<std::string> SplitLine(const std::string& a) {
  if (!a.size()) return std::vector<std::string>();
  bool flag1 = false, flag2 = false;
  std::vector<std::string> res(1);
  for (char i : a) {
    if (flag1) {
      if (i == '\"') flag1 = false, flag2 = true;
      else res.back().push_back(i);
    }
    else {
      if (i == '\"') {
        flag1 = true;
        if (flag2) flag2 = false, res.back().push_back('\"');
      }
      else if (i == ',') {
        res.push_back("");
        flag2 = false;
      }
      else res.back().push_back(i);
    }
  }
  return res;
}

// read a comma-separated csv file
std::vector<std::vector<std::string>> ReadCsv(const std::string& filename) {
  std::fstream fin(filename, std::ios::in);
  if (!fin) return std::vector<std::vector<std::string>>();

  std::string str, now;
  std::vector<std::vector<std::string>> res;
  bool flag = false;
  while (getline(fin, str)) {
    now += str;
    flag ^= std::count(str.begin(), str.end(), '\"') & 1;
    if (!flag) { // total '\"' character count is even - line completed
      res.emplace_back(SplitLine(now));
      now.clear();
    }
    else now += '\n';
  }
  return res;
}
