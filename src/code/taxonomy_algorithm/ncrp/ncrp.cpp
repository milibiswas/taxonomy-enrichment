//
// An example implementaiton of NetHiex.
//
// Usage:
//   ./ncrp edge-list.txt output.txt
//
//   edge-list.txt: should contain m lines, where m is the
//   number of edges. Each line contains two integer IDs,
//   meaning that there is an undirected edge between the two.
//
//   output.txt: contains the output, i.e., the node embeddings.
//
// Hyper-parameters are hard-coded (I'm sorry), see the comments in main().
// I'm not sure whether the default hyper-parameters here are the ones that
// were used when producing the reported results or not (most likely not).
// Personally I won't mind it if you don't put a lot of effort into
// tuning the hyper-parameters when using my algorithm as your baseline,
// as I know it can be painful.
//

#include <cassert>
#include <iostream>
#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dlib/optimization.h"

using namespace std;

#define BERNOULLI_DIST

#define NO_COPY_OR_MOVE(T)             \
  T(T const &) = delete;               \
  void operator=(T const &t) = delete; \
  T(T &&) = delete

typedef double Real;
typedef dlib::matrix<Real, 0, 1> ColVec;

Real const PI = 3.14159265359;
Real const EPS = 1e-6;

class RNG {
 public:
  RNG() { _eng.seed(random_device()()); }

  explicit RNG(int seed) : _eng(seed) {}

  Real normal(Real mean = 0.0, Real stddev = 1.0) {
    return _normal_d(_eng, normal_distribution<>::param_type(mean, stddev));
  }

  int unifInt(int min_included, int max_included) {
    return _unif_int_d(_eng, uniform_int_distribution<>::param_type(
                                 min_included, max_included));
  }

  Real unifReal(Real min_included = 0.0, Real max_included = 1.0) {
    return _unif_real_d(_eng, uniform_real_distribution<>::param_type(
                                  min_included, max_included));
  }

  int poisson(Real lambda) {
    return _poisson_d(_eng, poisson_distribution<>::param_type(lambda));
  }

 private:
  mt19937 _eng;
  normal_distribution<> _normal_d;
  uniform_int_distribution<> _unif_int_d;
  uniform_real_distribution<> _unif_real_d;
  poisson_distribution<> _poisson_d;

  NO_COPY_OR_MOVE(RNG);
};

RNG rng;

class AliasSampler {
 public:
  explicit AliasSampler(vector<Real> const &probs) {
    Real probs_sum = accumulate(probs.begin(), probs.end(), 0.0);
    int sz = probs.size();
    assert(sz > 0);
    _quot.resize(sz);
    _jump.resize(sz);
    iota(_jump.begin(), _jump.end(), 0);
    queue<int> smaller, larger;
    for (int i = 0; i < sz; i++) {
      _quot[i] = sz * probs[i] / probs_sum;
      if (_quot[i] < 1.0)
        smaller.push(i);
      else
        larger.push(i);
    }
    while (!smaller.empty() && !larger.empty()) {
      int small = smaller.front();
      smaller.pop();
      int large = larger.front();
      larger.pop();
      _jump[small] = large;
      _quot[large] -= (1.0 - _quot[small]);
      if (_quot[large] < 1.0)
        smaller.push(large);
      else
        larger.push(large);
    }
  }

  int draw() const {
    int sz = _jump.size();
    int i = rng.unifInt(0, sz - 1);
    if (rng.unifReal() < _quot[i]) return i;
    return _jump[i];
  }

 private:
  vector<Real> _quot;
  vector<int> _jump;

  NO_COPY_OR_MOVE(AliasSampler);
};

class Graph {
 public:
  explicit Graph(vector<pair<int, int>> const &edge_list,
                 bool directed = false) {
    _n = 0;
    _edge_sets.clear();
    for (pair<int, int> e : edge_list) {
      int i = e.first, j = e.second;
      _n = max<int>(_n, i + 1);
      _n = max<int>(_n, j + 1);
      _edge_sets.resize(_n);
      _edge_sets[i].insert(j);
      if (!directed) _edge_sets[j].insert(i);
    }
    _edges.clear();
    _edges.resize(_n);
    for (int i = 0; i < _n; i++)
      for (int j : _edge_sets[i]) _edges[i].push_back(j);
  }

  void setupTransTables(Real p, Real q) {
    assert(p > EPS && q > EPS);
    _transp.clear();
    _transp.resize(_n);
    cout << "Building transition tables... " << flush;
    for (int u = 0; u < _n; u++)
      for (int v : _edges[u]) {
        vector<Real> pbs(_edges[v].size());
        for (int k = 0; k < (int)_edges[v].size(); k++) {
          int w = _edges[v][k];
          if (w == u)
            pbs[k] = 1.0 / p;
          else if (_edge_sets[u].count(w) > 0)
            pbs[k] = 1.0;
          else
            pbs[k] = 1.0 / q;
        }
        assert(_transp[u].count(v) == 0);
        _transp[u].emplace(v, pbs);
      }
    cout << "Done." << endl;
  }

  void freeTrans() {
    _transp.clear();
    _transp.shrink_to_fit();
  }

  int numNodes() const { return _n; }

  void randWalks(vector<unordered_map<int, Real>> &cooc_cnts, Real p = 1.0,
                 Real q = 1.0, int walks_per_src = 10, int walk_len = 20,
                 int win_sz = 6) {
    cout << "Random walking... " << flush;
    bool non_bias = (fabs(p - 1.0) < EPS && fabs(q - 1.0) < EPS);
    if (!non_bias && _transp.empty()) setupTransTables(p, q);
    cooc_cnts.resize(_n);
    vector<int> path;
    for (int _ = 0; _ < walks_per_src; _++)
      for (int u = 0; u < _n; u++) {
        path.clear();
        path.push_back(u);
        while ((int)path.size() < walk_len) {
          int cur = path.back();
          if (_edges[cur].empty()) break;
          int nxt;
          if (non_bias || path.size() == 1)
            nxt = _edges[cur][rng.unifInt(0, (int)_edges[cur].size() - 1)];
          else
            nxt = _edges[cur][_transp[path[path.size() - 2]].at(cur).draw()];
          path.push_back(nxt);
        }
        for (int i = 0; i < (int)path.size(); i++) {
          int beg = max<int>(i - win_sz / 2, 0);
          int end = min<int>(i + win_sz / 2, (int)path.size() - 1);
          int u = path[i];
          for (int j = beg; j <= end; j++) {
            int v = path[j];
            if (cooc_cnts[u].count(v) == 0)
              cooc_cnts[u][v] = 1.0;
            else
              cooc_cnts[u][v] += 1.0;
          }
        }
      }
    cout << "Done." << endl;
  }

  void negSample(vector<unordered_map<int, Real>> &cooc_cnts) {
    cout << "Negative sampling... " << flush;
    cooc_cnts.resize(_n);
    vector<Real> prb(_n);
    for (int u = 0; u < _n; u++) prb[u] = pow((Real)_edges[u].size(), 0.75);
    AliasSampler smplr(prb);
    for (int u = 0; u < _n; u++) {
      int n_neg = 0.0;
      for (pair<int, Real> v_c : cooc_cnts[u]) n_neg += v_c.second;
      for (int _ = 0; _ < n_neg; _++) {
        int v = smplr.draw();
        if (cooc_cnts[u].count(v) == 0)
          cooc_cnts[u][v] = -1.0;
        else
          cooc_cnts[u][v] -= 1.0;
      }
    }
    cout << "Done." << endl;
  }

 private:
  int _n;
  vector<unordered_set<int>> _edge_sets;
  vector<vector<int>> _edges;
  vector<unordered_map<int, AliasSampler>> _transp;

  NO_COPY_OR_MOVE(Graph);
};

Real ln_nrml(Real x) { return -0.5 * x * x - 0.5 * log(2.0 * PI); }

class Model {
 public:
  Model(int N, int D, int L, Real beta_v, Real sgm_x, Real sgm_xy)
      : _N(N),
        _D(D),
        _L(L),
        _D_node(D / (L + 1)),
        _beta_v(beta_v),
        _sgm_x(sgm_x),
        _sgm_xy(sgm_xy) {
    assert(N > 1);
    assert(D > L);
    assert(L > 1);
    assert(beta_v >= 1.0);  // Blei (2003) et al use 1.0.
    assert(sgm_x > EPS);
    assert(sgm_xy > EPS);
    assert(_D_node > 0);
    assert(D - _D_node * L > 0);
    init();
  }

  void init() {
    /* Initialize the tree. */
    _tree_sz = _L;
    _ch_ls.clear();
    _ch_ls.resize(_tree_sz);
    for (int i = 0; i < _tree_sz - 1; i++) _ch_ls[i].push_back(i + 1);
    _lv.clear();
    for (int i = 0; i < _tree_sz; i++) _lv.push_back(i);
    _v.clear();
    _v.resize(_tree_sz, 1.0 / (1.0 + _beta_v));
    _v[0] = 1.0;
    _w.clear();
    _w.resize(_tree_sz, vector<Real>(_D_node, 0.0));

    /* Initialize representations. */
    _x.resize(_N, vector<Real>(_D));
    for (int i = 0; i < _N; i++)
      for (int k = 0; k < _D; k++) _x[i][k] = rng.normal();
  }

  void makeData(Graph &g) {
    static vector<unordered_map<int, Real>> cooc;
    cooc.clear();
    g.randWalks(cooc);
    g.freeTrans();
    g.negSample(cooc);
    assert((int)cooc.size() == _N);
    _r.clear();
    _r.resize(_N);
    for (int u = 0; u < _N; u++)
      for (pair<int, Real> v_c : cooc[u])
        if (v_c.first != u && fabs(v_c.second) > EPS)  // Don't keep loops.
#ifdef BERNOULLI_DIST
          _r[u].push_back(v_c);
#else
          _r[u].push_back(
              make_pair(v_c.first, 1.0 / (1.0 + exp(-v_c.second * 6.0))));
#endif
  }

  void iterate() {
    checkSanity();
    runExpStep();
    runMaxStep();
    runRefStep();
  }

  vector<vector<Real>> const &getRepr() const { return _x; }

 private:
  void checkSanity() {
    assert(_L > 0);
    assert(_tree_sz >= _L);
    assert((int)_ch_ls.size() == _tree_sz);
    assert((int)_lv.size() == _tree_sz);
    assert((int)_v.size() == _tree_sz);
    assert((int)_w.size() == _tree_sz);
    static vector<bool> is_child;
    is_child.clear();
    is_child.resize(_tree_sz, false);
    for (int t = 0; t < _tree_sz; t++) {
      if (_lv[t] < _L - 1)
        assert((int)_ch_ls[t].size() > 0);
      else
        assert(_ch_ls[t].empty());
      assert(0 <= _lv[t] && _lv[t] < _L);
      assert(-EPS < _v[t] && _v[t] < 1.0 + EPS);
      assert((int)_w[t].size() == _D_node);
      int last_ch = t;
      for (int ch : _ch_ls[t]) {
        assert(last_ch < ch);
        last_ch = ch;
        assert(0 <= ch && ch < _tree_sz);
        assert(_lv[ch] == _lv[t] + 1);
        is_child[ch] = true;
      }
    }
    assert(!is_child[0]);
    assert(_lv[0] == 0 && _lv[0] < _L - 1);
    int n_paths = 0;
    for (int t = 1; t < _tree_sz; t++) {
      assert(is_child[t]);
      if (_lv[t] == _L - 1) n_paths += 1;
    }
  }

  void runExpStep() {
    Real ln_sx = log(_sgm_x);

    /* Compute log p(c | V). */
    static vector<Real> ln_prb_c;
    ln_prb_c.clear();
    ln_prb_c.resize(_tree_sz);
    ln_prb_c[0] = 0.0;
    for (int t = 0; t < _tree_sz; t++) {
      Real ln_prod_1mv = 0.0;
      for (int ch : _ch_ls[t]) {
        ln_prb_c[ch] = ln_prb_c[t] + ln_prod_1mv + log(_v[ch]);
        ln_prod_1mv += log(1.0 - _v[ch]);
      }
      ln_prb_c[t] += ln_prod_1mv;  // Probability of paths NOT in the tree.
    }

    /* Compute q(c_i). */
    _q.resize(_N);
    _q_sum.resize(_N);
    static vector<Real> ln_p_x_w0, ln_p_x_wt;
    ln_p_x_w0.resize(_L);
    ln_p_x_wt.resize(_tree_sz);
    for (int i = 0; i < _N; i++) {
      vector<Real> const &xi = _x[i];

      /* Compute log p(x_i | w_c = 0). */
      ln_p_x_w0[_L - 1] = 0.0;
      for (int lv = _L - 2; lv >= 0; lv--) {
        Real ln_p = 0.0;
        for (int k = 0; k < _D_node; k++)
          ln_p += ln_nrml(xi[(lv + 1) * _D_node + k] / _sgm_x) - ln_sx;
        ln_p_x_w0[lv] = ln_p + ln_p_x_w0[lv + 1];
      }

      /* Compute log p(x_i | w_c). */
      ln_p_x_wt[0] = 0.0;
      for (int k = 0; k < _D_node; k++)
        ln_p_x_wt[0] += ln_nrml((xi[k] - _w[0][k]) / _sgm_x) - ln_sx;
      for (int t = 0; t < _tree_sz; t++)
        for (int ch : _ch_ls[t]) {
          int lv = _lv[ch];
          Real ln_p = 0.0;
          for (int k = 0; k < _D_node; k++)
            ln_p +=
                ln_nrml((xi[lv * _D_node + k] - _w[ch][k]) / _sgm_x) - ln_sx;
          ln_p_x_wt[ch] = ln_p + ln_p_x_wt[t];
        }

      /* Compute q(c_i) ~ p(c | V) p(x_i | w_c). */
      vector<Real> &qi = _q[i];
      qi.resize(_tree_sz);
      for (int t = 0; t < _tree_sz; t++)
        qi[t] = ln_prb_c[t] + ln_p_x_wt[t] + ln_p_x_w0[_lv[t]];
      Real cmax = qi[0];
      for (int t = 0; t < _tree_sz; t++) cmax = max<Real>(cmax, qi[t]);
      Real prb_sum = 0.0;
      for (int t = 0; t < _tree_sz; t++) {
        qi[t] = exp(qi[t] - cmax);
        prb_sum += qi[t];
      }
      for (int t = 0; t < _tree_sz; t++) qi[t] /= prb_sum;
      vector<Real> &qi_sum = _q_sum[i];
      qi_sum.resize(_tree_sz);
      for (int t = _tree_sz - 1; t >= 0; t--) {
        qi_sum[t] = qi[t];
        for (int ch : _ch_ls[t]) qi_sum[t] += qi_sum[ch];
      }
      assert(1.0 - EPS < qi_sum[0] && qi_sum[0] < 1.0 + EPS);
    }

    _q_all.clear();
    _q_all.resize(_tree_sz, 0.0);
    for (vector<Real> const &qi_sum : _q_sum)
      for (int t = 0; t < _tree_sz; t++) _q_all[t] += qi_sum[t];
  }

  void runMaxStep() {
    runMaxStepXY();
    runMaxStepW();
    runMaxStepV();
  }

  void runMaxStepW() {
    for (int k = 0; k < _D_node; k++) _w[0][k] = 0.0;
    for (int t = 1; t < _tree_sz; t++)
      for (int k = 0; k < _D_node; k++) {
        Real a = 0.0, b = 0.0;
        int kk = _lv[t] * _D_node + k;
        for (int i = 0; i < _N; i++) {
          Real qi = _q_sum[i][t];
          a += qi * _x[i][kk];
          b += qi;
        }
        _w[t][k] = ((b <= EPS) ? 0.0 : (a / b));
      }
  }

  void runMaxStepV() {
    _v[0] = 1.0;
    for (int t = 0; t < _tree_sz; t++) {
      Real neg = _beta_v - 1.0;
      for (vector<Real> const &qi : _q) neg += qi[t];
      for (int ch_k = (int)_ch_ls[t].size() - 1; ch_k >= 0; ch_k--) {
        int ch = _ch_ls[t][ch_k];
        Real pos = _q_all[ch];
        assert(pos + neg > EPS);
        if (pos + neg > EPS)
          _v[ch] = pos / (pos + neg);
        else
          _v[ch] = 1.0 / (1.0 + _beta_v);
        // The following two lines are for numerical stability.
        _v[ch] = max<Real>(_v[ch], EPS);        // About log(_v[ch]).
        _v[ch] = min<Real>(_v[ch], 1.0 - EPS);  // About log(1 - _v[ch]).
        neg += pos;
      }
    }
  }

  void runMaxStepXY() {
    int D = _D, L = _L, D_node = _D_node;
    Real sx2 = pow(_sgm_x, 2), sxy2 = pow(_sgm_xy, 2);
    vector<int> const &lv = _lv;
    vector<vector<Real>> const &w = _w;
    vector<vector<Real>> const &y = _x;
    Real loss = 0.0;
    auto opt_cfg = make_pair(dlib::cg_search_strategy(),
                             dlib::objective_delta_stop_strategy(1e-6, 1));

    /* Optimize _x[i]. */
    for (int i = 0; i < _N; i++) {
      static vector<Real> qi_w0;
      qi_w0.clear();
      qi_w0.resize(_L, 0.0);
      for (int t = 0; t < _tree_sz; t++)
        for (int k = _lv[t] + 1; k < _L; k++) qi_w0[k] += -_q[i][t] / sx2;
      vector<Real> const &qi_sum = _q_sum[i];
      vector<pair<int, Real>> const &ri = _r[i];

      auto fx = [D, L, D_node, sx2, sxy2, &lv, &w, &y, &qi_sum,
                 &ri](ColVec const &xi) {
        Real val = 0.0;

        /* Loss of p(x | w). */
        int tree_sz = qi_sum.size();
        for (int t = 0; t < tree_sz; t++) {
          Real c = -0.5 * qi_sum[t] / sx2;
          for (int k = 0; k < D_node; k++)
            val += c * pow(xi(D_node * lv[t] + k) - w[t][k], 2);
        }
        for (int i = 1; i < L; i++)
          for (int k = 0; k < D_node; k++)
            val += 0.5 * qi_w0[i] * pow(xi(D_node * i + k), 2);

        /* Loss of p(r | x, y). */
        Real cnt_samples = 0.0;
        for (pair<int, Real> const &j_rij : ri) {
          int j = j_rij.first;
          Real rij = j_rij.second;
#ifdef BERNOULLI_DIST
          /* Bernoulli distribution. */
          Real ds = 0.0;
          for (int k = 0; k < D; k++) ds += pow(xi(k) - y[j][k], 2);
          if (rij > 0.0)
            val += -ds / sxy2 * rij;
          else
            val += log(1.0 - exp(-ds / sxy2)) * -rij;
          cnt_samples += fabs(rij);
#else
          /* Normal distribution. */
          Real xy = 0.0;
          for (int k = 0; k < D; k++) xy += xi(k) * y[j][k];
          val += -0.5 * pow(xy - rij, 2) / sxy2;
          cnt_samples += 1.0;
#endif
        }
        val *= (sxy2 / (cnt_samples + 1.0));
        return val;
      };
      auto dx = [D, L, D_node, sx2, sxy2, &lv, &w, &y, &qi_sum,
                 &ri](ColVec const &xi) {
        ColVec dx(D);
        for (int k = 0; k < D; k++) dx(k) = 0.0;

        /* Gradient of p(x | w). */
        int tree_sz = qi_sum.size();
        for (int t = 0; t < tree_sz; t++) {
          Real c = -qi_sum[t] / sx2;
          for (int k = 0; k < D_node; k++) {
            int kk = D_node * lv[t] + k;
            dx(kk) += c * (xi(kk) - w[t][k]);
          }
        }
        for (int i = 1; i < L; i++)
          for (int k = 0; k < D_node; k++) {
            int kk = D_node * i + k;
            dx(kk) += qi_w0[i] * xi(kk);
          }

        /* Gradient of p(x | w). */
        Real cnt_samples = 0.0;
        for (pair<int, Real> const &j_rij : ri) {
          int j = j_rij.first;
          Real rij = j_rij.second;
#ifdef BERNOULLI_DIST
          /* Bernoulli distribution. */
          if (rij > 0.0) {
            for (int k = 0; k < D; k++)
              dx(k) += 2.0 * (y[j][k] - xi(k)) / sxy2 * rij;
          } else {
            Real ds = 0.0;
            for (int k = 0; k < D; k++) ds += pow(xi(k) - y[j][k], 2);
            Real c = 2.0 / (exp(ds / sxy2) - 1.0) / sxy2 * -rij;
            for (int k = 0; k < D; k++) dx(k) += c * (xi(k) - y[j][k]);
          }
          cnt_samples += fabs(rij);
#else
          /* Normal distribution. */
          Real xy = 0.0;
          for (int k = 0; k < D; k++) xy += xi(k) * y[j][k];
          for (int k = 0; k < D; k++) dx(k) += -(xy - rij) / sxy2 * y[j][k];
          cnt_samples += 1.0;
#endif
        }
        Real c = sxy2 / (cnt_samples + 1.0);
        for (int k = 0; k < D; k++) dx(k) *= c;
        return dx;
      };
      static ColVec xi(_D);
      for (int k = 0; k < _D; k++) xi(k) = _x[i][k];
      loss += dlib::find_max(opt_cfg.first, opt_cfg.second, fx, dx, xi, 0.0);
      for (int k = 0; k < _D; k++) _x[i][k] = xi(k);
    }
    cout << "loss = " << -loss / _N << endl;
  }

  void plotTree(int t = 0) {
    for (int _ = 0; _ < _lv[t]; _++) cout << "    ";
    cout << t << "``";
    if (_lv[t] < _L - 2) {
      cout << "\\" << endl;
      for (int ch : _ch_ls[t]) plotTree(ch);
    } else if (_lv[t] == _L - 2) {
      cout << "`";
      for (int ch : _ch_ls[t]) cout << ch << "(" << int(_q_all[ch]) << ") ";
      cout << endl;
    }
  }

  void runRefStep(int max_paths = 10) {
    static vector<int> leaves;
    leaves.clear();
    for (int t = 0; t < _tree_sz; t++)
      if (_lv[t] == _L - 1) leaves.push_back(t);
    cout << leaves.size() << " paths before adjustment." << endl;
    plotTree(0);

    Real avg_branch = 0.0;
    for (int t = 0; t < _tree_sz; t++) avg_branch += _ch_ls[t].size();
    avg_branch /= (_tree_sz - leaves.size());
    cout << "average branching factor: " << avg_branch << endl;

    static vector<bool> to_del;
    to_del.clear();
    to_del.resize(_tree_sz, false);

    /* Prune paths that are visited by less than x% of the samples. */
    int cnt_prune_delta = 0;
    for (int leaf : leaves)
      if (_q_all[leaf] / (Real)_N < 0.01) {
        to_del[leaf] = true;
        cnt_prune_delta++;
      }
    cout << cnt_prune_delta << " rarely visited paths." << endl;

    /* Prune redundant paths. */
    int cnt_prune_corr = 0;
    for (int a = 0; a < (int)leaves.size() - 1; a++)
      if (!to_del[leaves[a]])
        for (int b = a + 1; b < (int)leaves.size(); b++)
          if (!to_del[leaves[b]]) {
            int ch_a = leaves[a], ch_b = leaves[b];
            Real corr = 0.0, nm_a = 0.0, nm_b = 0.0;
            for (int i = 0; i < _N; i++) {
              corr += _q[i][ch_a] * _q[i][ch_b];
              nm_a += pow(_q[i][ch_a], 2);
              nm_b += pow(_q[i][ch_b], 2);
            }
            corr /= (sqrt(nm_a) * sqrt(nm_b));
            if (corr > 0.95) {
              to_del[ch_b] = true;
              cnt_prune_corr++;
            }
          }
    cout << cnt_prune_corr << " redundant paths." << endl;

    /* Grow by sampling new paths from inner nodes. */
    int n_del = cnt_prune_corr + cnt_prune_delta;
    int n_new = min<int>(rng.poisson(1), max_paths - leaves.size() + n_del);
    if (n_del == (int)leaves.size()) n_new = max<int>(n_new, 1);
    cout << n_new << " new paths to explore." << endl;
    static vector<Real> prb_inn;
    prb_inn.clear();
    for (int t = 0; t < _tree_sz; t++)
      if (_lv[t] < _L - 1) {
        Real p = 0.0;
        for (int i = 0; i < _N; i++) p += _q[i][t];
        assert(t == (int)prb_inn.size());
        prb_inn.push_back(p);
      }
    AliasSampler smp(prb_inn);
    for (int _ = 0; _ < n_new; _++) {
      int t = smp.draw();
      assert(_lv[t] < _L - 1);
      assert(_tree_sz == (int)_lv.size());
      assert(_tree_sz == (int)_ch_ls.size());
      assert(_tree_sz == (int)_v.size());
      assert(_tree_sz == (int)_w.size());
      _ch_ls[t].push_back(_tree_sz);
      for (int lv = _lv[t] + 1; lv < _L; lv++) {
        _lv.push_back(lv);
        _ch_ls.push_back(vector<int>());
        if (lv < _L - 1)
          _ch_ls[_tree_sz].push_back(_tree_sz + 1);
        else
          leaves.push_back(_tree_sz);
        _v.push_back(1.0 / (1.0 + _beta_v));
        _w.push_back(vector<Real>(_D_node, 0.0));
        to_del.push_back(false);
        _tree_sz++;
      }
    }

    /* Invalidate temporary data. */
    _q.clear();
    _q_sum.clear();
    _q_all.clear();

    /*  Ensure the sanity of the tree. */
    for (int t = _tree_sz - 1; t >= 0; t--)
      if (!_ch_ls[t].empty()) {
        assert(!to_del[t]);
        bool all_del = true;
        for (int ch : _ch_ls[t])
          if (!to_del[ch]) {
            all_del = false;
            break;
          }
        to_del[t] = all_del;
      }
    assert(to_del[0] == false);
    static vector<int> old2new;
    old2new.clear();
    old2new.resize(_tree_sz, -1);
    old2new[0] = 0;
    static vector<int> new2old;
    new2old.clear();
    new2old.push_back(0);
    static vector<int> new_lv;
    new_lv.clear();
    new_lv.push_back(0);
    static vector<vector<int>> new_chls;
    new_chls.clear();
    static vector<Real> new_v;
    new_v.clear();
    static vector<vector<Real>> new_w;
    new_w.clear();
    for (int new_t = 0; new_t < (int)new2old.size(); new_t++) {
      int t = new2old[new_t];
      int lv = new_lv[new_t];
      assert(lv == _lv[t]);
      new_chls.push_back(vector<int>());
      for (int ch : _ch_ls[t])
        if (!to_del[ch]) {
          int new_ch = new2old.size();
          old2new[ch] = new_ch;
          new2old.push_back(ch);
          new_lv.push_back(lv + 1);
          new_chls[new_t].push_back(new_ch);
        }
      new_v.push_back(_v[t]);
      new_w.push_back(_w[t]);
    }
    _tree_sz = new2old.size();
    _lv.swap(new_lv);
    _ch_ls.swap(new_chls);
    _v.swap(new_v);
    _w.swap(new_w);
  }

  /* Observed constants. */
  int const _N;                        // Number of nodes.
  vector<vector<pair<int, Real>>> _r;  // Observed tuples: (i, j, r_ij).

  /* Hyper-parameters. */
  int const _D;        // Number of dimensions.
  int const _L;        // Level of the tree.
  int const _D_node;   // D_node = D / (L + 1).
  Real const _beta_v;  // Stick-breaking process: v ~ Beta(1, beta_v).
  Real const _sgm_x;   // x ~ Normal(w, sigma_x).
  Real const _sgm_xy;  // r_ij ~ Normal(<x_i, y_j>, sigma_xy).

  /* The tree structure. */
  int _tree_sz;
  vector<vector<int>> _ch_ls;  // ID_top < ID_bottom, ID_left < ID_right.
  vector<int> _lv;

  /* Latent variables to be estimated. */
  vector<Real> _v;  // v[0] is always one.
  vector<vector<Real>> _w;
  vector<vector<Real>> _q;
  vector<vector<Real>> _q_sum;
  vector<Real> _q_all;  // \sum_{i=1,2,...,N} _q_sum[i].
  vector<vector<Real>> _x;

  NO_COPY_OR_MOVE(Model);
};

int main(int argc, char *argv[]) {
  assert(argc == 3);
  vector<pair<int, int>> edges;
  {
    ifstream fin(argv[1]);
    int u, v;
    while (fin >> u) {
      fin >> v;
      assert(u >= 0 && v >= 0);
      edges.push_back(make_pair(u, v));
    }
  }
  Graph g(edges);
  edges.clear();
  edges.shrink_to_fit();
  int N = g.numNodes();
  int D = 128;  // Hyper-parameter: Dimension of embeddings. I'd recommend using a larger one.
  Model model(N, D, 4, 1.0, 0.50, 2.00);  // Hyper-parameters: layers, beta_v, sigma_x, sigma_xy.
  int T = 20;  // Hyper-parameter: number of training iterations.
  for (int iter = 0; iter < T; iter++) {
    cout << "[" << iter << "/" << T << "]" << endl;
    model.makeData(g);
    model.iterate();
    cout << endl;
  }
  {
    ofstream fout(argv[2]);
    vector<vector<Real>> const &feats = model.getRepr();
    fout << feats.size() << " " << feats[0].size() << endl;
    for (int i = 0; i < N; i++) {
      fout << i;
      for (int d = 0; d < D; d++) fout << " " << feats[i][d];
      fout << endl;
    }
  }
  return 0;
}
