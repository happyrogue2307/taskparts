#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <type_traits>
#include <utility>

#include <taskparts/nativeforkjoin.hpp>

// **************************************************************
// Parallel Delaunay Triangulation
// From the paper:
// Randomized Incremental Convex Hull is Highly Parallel
// Blelloch, Gu, Shun and Sun
//
// Port of parlaylib's examples/delaunay.h to taskparts' native
// fork-join fibers. Every fork goes through taskparts::fork2join.
// The parlay pieces that the example relies on (par_do3,
// parallel_for, tabulate, pack, merge, the parallel loops inside
// parlay::sequence, and examples/hash_map.h) are reimplemented
// below following parlay's own algorithms and grain sizes, so that
// the fork tree matches parlay's as closely as possible. Memory
// comes from malloc instead of parlay's pool allocator.
// **************************************************************

namespace delaunay_nativefj {

using real = float;
using point_id = int;

struct point {
  point_id id; real x; real y;
  const bool operator<(const point p) const {return id < p.id;}
};

using tri = std::array<point_id,3>;
using edge = std::array<point_id,2>;

// A "staged" version of in_cicle.
// It takes 3 points, returning a function on one more point that
// reports whether that point is in the circumcircle of the points.
// Since we often check multiple points against a single circle
// this reduces the work.
// Projects onto a parobola and does a planeside test.
inline auto in_circle (point a, point b, point d) {
  struct vect { double x,y,z;};
  auto project = [=] (point p) {
    double px = p.x-d.x; double py = p.y-d.y;
    return vect{px, py, px*px + py*py};};
  auto cross = [] (vect v1, vect v2) {
    return vect{v1.y*v2.z - v1.z*v2.y,
                v1.z*v2.x - v1.x*v2.z,
                v1.x*v2.y - v1.y*v2.x};};
  vect cp = cross(project(a), project(b));
  return [=] (point c) -> bool{
    auto dot = [] (vect v1, vect v2) {
      return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;};
    return dot(cp, project(c)) > 0.0;};
}

/*---------------------------------------------------------------------*/
/* Hashing and random numbers (from parlay/utilities.h, parlay/random.h) */

inline uint64_t hash64(uint64_t u) {
  uint64_t v = u * 3935559000370003845ul + 2691343689449507681ul;
  v ^= v >> 21;
  v ^= v << 37;
  v ^= v >> 4;
  v *= 4768777513237032717ul;
  v ^= v << 20;
  v ^= v >> 41;
  v ^= v << 5;
  return v;
}

// Combine two hash values into a single hash value. Formula borrowed from Boost
inline void hash_combine(size_t& seed, size_t v) {
  seed ^= v + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <typename T>
struct hash {
  static_assert(std::is_integral_v<T>);
  size_t operator()(const T& p) const {
    return p * UINT64_C(0xbf58476d1ce4e5b9);
  }
};

template <typename V, std::size_t n>
struct hash<std::array<V,n>> {
  size_t operator()(const std::array<V,n>& a) const {
    size_t h = 1;
    for (int i=0; i < n; i++) {
      hash_combine(h, hash<V>{}(a[i]));
    }
    return h;
  }
};

// Same generator as parlay::random_generator, so that parlay and
// taskparts runs can be given identical inputs
struct random_generator {
 public:
  using result_type = size_t;
  explicit random_generator(size_t seed) : state(seed) { }
  random_generator() : state(0) { }
  void seed(result_type value = 0) { state = value; }
  result_type operator()() { return state = hash64(state); }
  static constexpr result_type max() { return std::numeric_limits<result_type>::max(); }
  static constexpr result_type min() { return std::numeric_limits<result_type>::lowest(); }
  random_generator operator[](size_t i) const {
    return random_generator(static_cast<size_t>(hash64((i+1)*0x7fffffff + state))); }
 private:
  result_type state = 0;
};

/*---------------------------------------------------------------------*/
/* Parlay-style primitives and the triangulation, all forking through
 * taskparts::fork2join on the given Scheduler */

template <typename Scheduler>
struct fj {

  /*-------------------------------------------------------------------*/
  /* Fork join (parlay/parallel.h, parlay/scheduler.h) */

  // left runs on the current worker, right may be stolen (as in parlay)
  template <typename Lf, typename Rf>
  static void par_do(const Lf& left, const Rf& right) {
    taskparts::fork2join(left, right, Scheduler());
  }

  template <typename Lf, typename Mf, typename Rf>
  static void par_do3(const Lf& left, const Mf& mid, const Rf& right) {
    auto left_mid = [&] { par_do(left, mid); };
    par_do(left_mid, right);
  }

  // Grow a sequential prefix of the loop until it takes at least 1us,
  // and return the number of iterations it ran
  template <typename F>
  static size_t get_granularity(size_t start, size_t end, const F& f) {
    size_t done = 0;
    size_t sz = 1;
    uint64_t ticks = 0;
    do {
      sz = std::min(sz, end - (start + done));
      auto tstart = std::chrono::steady_clock::now();
      for (size_t i = 0; i < sz; i++) f(start + done + i);
      auto tstop = std::chrono::steady_clock::now();
      ticks = static_cast<uint64_t>(std::chrono::duration_cast<
                std::chrono::nanoseconds>(tstop - tstart).count());
      done += sz;
      sz *= 2;
    } while (ticks < 1000 && done < (end - start));
    return done;
  }

  template <typename F>
  static void parfor_(size_t start, size_t end, const F& f, size_t granularity) {
    if ((end - start) <= granularity) {
      for (size_t i = start; i < end; i++) f(i);
    } else {
      size_t n = end - start;
      // Not in middle to avoid clashes on set-associative caches on powers of 2.
      size_t mid = (start + (9 * (n + 1)) / 16);
      par_do([&] { parfor_(start, mid, f, granularity); },
             [&] { parfor_(mid, end, f, granularity); });
    }
  }

  // A granularity of 0 picks one automatically, as in parlay
  template <typename F>
  static void parallel_for(size_t start, size_t end, const F& f, size_t granularity = 0) {
    if (start + 1 == end) {
      f(start);
    } else if ((end - start) <= granularity) {
      for (size_t i = start; i < end; i++) f(i);
    } else if (end > start) {
      if (granularity == 0) {
        size_t done = get_granularity(start, end, f);
        granularity = std::max(done, (end - start) / (128 * taskparts::perworker::nb_workers()));
        start += done;
      }
      parfor_(start, end, f, granularity);
    }
  }

  /*-------------------------------------------------------------------*/
  /* Sequences (parlay/sequence.h): a buffer whose copy, initialization
   * and destruction loops are parallel, with parlay's grain sizes */

  template <typename T>
  class sequence {
  public:

    using value_type = T;

    sequence() { }

    // n value-initialized elements
    explicit sequence(size_t n) : buf(allocate(n)), n(n) {
      auto b = buf;
      parallel_for(0, n, [&] (size_t i) { new (&b[i]) T(); }, initialization_granularity());
    }

    sequence(std::initializer_list<T> l) : buf(allocate(l.size())), n(l.size()) {
      std::uninitialized_copy(l.begin(), l.end(), buf);
    }

    sequence(const sequence& other) : buf(allocate(other.n)), n(other.n) {
      auto b = buf;
      auto o = other.buf;
      parallel_for(0, n, [&] (size_t i) { new (&b[i]) T(o[i]); }, copy_granularity());
    }

    sequence(sequence&& other) noexcept : buf(other.buf), n(other.n) {
      other.buf = nullptr;
      other.n = 0;
    }

    // copies from an lvalue, moves from an rvalue
    sequence& operator=(sequence other) noexcept {
      std::swap(buf, other.buf);
      std::swap(n, other.n);
      return *this;
    }

    ~sequence() {
      if (buf == nullptr) {
        return;
      }
      if constexpr (! std::is_trivially_destructible_v<T>) {
        auto b = buf;
        parallel_for(0, n, [&] (size_t i) { b[i].~T(); });
      }
      std::free(buf);
    }

    // n elements that the caller must construct (trivially copyable T only)
    static sequence uninitialized(size_t n) {
      static_assert(std::is_trivially_copyable_v<T>);
      sequence s;
      s.buf = allocate(n);
      s.n = n;
      return s;
    }

    template <typename F>
    static sequence from_function(size_t n, const F& f, size_t granularity = 0) {
      sequence s;
      s.buf = allocate(n);
      s.n = n;
      auto b = s.buf;
      parallel_for(0, n, [&] (size_t i) { new (&b[i]) T(f(i)); }, granularity);
      return s;
    }

    size_t size() const { return n; }
    T* data() { return buf; }
    const T* data() const { return buf; }
    T& operator[](size_t i) { return buf[i]; }
    const T& operator[](size_t i) const { return buf[i]; }
    T* begin() { return buf; }
    T* end() { return buf + n; }
    const T* begin() const { return buf; }
    const T* end() const { return buf + n; }

  private:

    T* buf = nullptr;
    size_t n = 0;

    static T* allocate(size_t n) {
      return (n == 0) ? nullptr : static_cast<T*>(std::malloc(n * sizeof(T)));
    }

    static constexpr size_t copy_granularity() {
      return (std::is_trivially_copyable_v<T>) ? (1 + (1024 * sizeof(size_t) / sizeof(T))) : 0;
    }

    static constexpr size_t initialization_granularity() {
      return (std::is_trivially_default_constructible_v<T>) ? (1 + (1024 * sizeof(size_t) / sizeof(T))) : 0;
    }

  };

  template <typename F>
  static auto tabulate(size_t n, const F& f, size_t granularity = 0) {
    using T = std::decay_t<std::invoke_result_t<const F&, size_t>>;
    return sequence<T>::from_function(n, f, granularity);
  }

  template <typename T>
  static sequence<T> append(const sequence<T>& A, const sequence<T>& B) {
    auto R = sequence<T>::uninitialized(A.size() + B.size());
    auto r = R.data();
    size_t nA = A.size();
    parallel_for(0, nA, [&] (size_t i) { r[i] = A[i]; });
    parallel_for(0, B.size(), [&] (size_t i) { r[nA + i] = B[i]; });
    return R;
  }

  /*-------------------------------------------------------------------*/
  /* Blocked loops, scan and pack (parlay/internal/sequence_ops.h) */

  static constexpr size_t log_block_size = 10;
  static constexpr size_t block_size = (1 << log_block_size);

  static size_t num_blocks(size_t n, size_t bs) {
    return (n == 0) ? 0 : (1 + ((n)-1) / (bs));
  }

  template <typename F>
  static void sliced_for(size_t n, size_t bs, const F& f) {
    size_t l = num_blocks(n, bs);
    auto body = [&] (size_t i) {
      size_t s = i * bs;
      size_t e = std::min(s + bs, n);
      f(i, s, e);
    };
    parallel_for(0, l, body, 1);
  }

  // exclusive plus-scan of A[s, e) into A, starting from offset; returns the total
  static size_t scan_serial(size_t* A, size_t s, size_t e, size_t offset) {
    size_t r = offset;
    for (size_t i = s; i < e; i++) {
      size_t t = A[i];
      A[i] = r;
      r += t;
    }
    return r;
  }

  // in-place exclusive plus-scan; returns the total
  static size_t scan_inplace(size_t* A, size_t n) {
    size_t l = num_blocks(n, block_size);
    if (l <= 2) {
      return scan_serial(A, 0, n, 0);
    }
    auto sums = sequence<size_t>::uninitialized(l);
    sliced_for(n, block_size, [&] (size_t i, size_t s, size_t e) {
      size_t r = 0;
      for (size_t j = s; j < e; j++) r += A[j];
      sums[i] = r;
    });
    size_t total = scan_serial(sums.data(), 0, l, 0);
    sliced_for(n, block_size, [&] (size_t i, size_t s, size_t e) {
      scan_serial(A, s, e, sums[i]);
    });
    return total;
  }

  // the elements of In whose flag in Fl is set, in order
  template <typename T>
  static sequence<T> pack(const sequence<T>& In, const sequence<bool>& Fl) {
    size_t n = In.size();
    size_t l = num_blocks(n, block_size);
    auto sum_bools = [&] (size_t s, size_t e) {
      size_t r = 0;
      for (size_t i = s; i < e; i++) r += Fl[i];
      return r;
    };
    auto pack_serial_at = [&] (size_t s, size_t e, T* out) {
      size_t k = 0;
      for (size_t i = s; i < e; i++) {
        if (Fl[i]) out[k++] = In[i];
      }
    };
    if (l == 1) {
      auto Out = sequence<T>::uninitialized(sum_bools(0, n));
      pack_serial_at(0, n, Out.data());
      return Out;
    }
    auto sums = sequence<size_t>::uninitialized(l);
    sliced_for(n, block_size, [&] (size_t i, size_t s, size_t e) {
      sums[i] = sum_bools(s, e);
    });
    size_t m = scan_inplace(sums.data(), l);
    auto Out = sequence<T>::uninitialized(m);
    sliced_for(n, block_size, [&] (size_t i, size_t s, size_t e) {
      pack_serial_at(s, e, Out.data() + sums[i]);
    });
    return Out;
  }

  /*-------------------------------------------------------------------*/
  /* Merge (parlay/internal/merge.h, parlay/internal/binary_search.h) */

  static constexpr size_t merge_base = 2000;

  static constexpr size_t binary_search_base = 16;

  // index of the first element of B[0, nB) that is not less than v
  template <typename T, typename Less>
  static size_t binary_search(const T* B, size_t nB, const T& v, const Less& less) {
    size_t start = 0;
    size_t end = nB;
    while (end - start > binary_search_base) {
      size_t mid = (end + start) / 2;
      if (! less(B[mid], v)) {
        end = mid;
      } else {
        start = mid + 1;
      }
    }
    for (size_t i = start; i < end; i++) {
      if (! less(B[i], v)) return i;
    }
    return end;
  }

  template <typename T, typename Less>
  static void seq_merge(const T* A, size_t nA, const T* B, size_t nB, T* R, const Less& f) {
    size_t i = 0;
    size_t j = 0;
    while (true) {
      if (i == nA) {
        while (j < nB) { R[i + j] = B[j]; j++; }
        break;
      }
      if (j == nB) {
        while (i < nA) { R[i + j] = A[i]; i++; }
        break;
      }
      if (f(B[j], A[i])) {
        R[i + j] = B[j];
        j++;
      } else {
        R[i + j] = A[i];
        i++;
      }
    }
  }

  template <typename T, typename Less>
  static void merge_into(const T* A, size_t nA, const T* B, size_t nB, T* R, const Less& f) {
    size_t nR = nA + nB;
    if (nR < merge_base) {
      seq_merge(A, nA, B, nB, R, f);
    } else if (nA == 0) {
      parallel_for(0, nB, [&] (size_t i) { R[i] = B[i]; });
    } else if (nB == 0) {
      parallel_for(0, nA, [&] (size_t i) { R[i] = A[i]; });
    } else {
      size_t mA = nA / 2;
      // important for stability that binary search identifies
      // first element in B greater or equal to A[mA]
      size_t mB = binary_search(B, nB, A[mA], f);
      if (mB == 0) mA++;  // ensures at least one on each side
      size_t mR = mA + mB;
      par_do([&] { merge_into(A, mA, B, mB, R, f); },
             [&] { merge_into(A + mA, nA - mA, B + mB, nB - mB, R + mR, f); });
    }
  }

  template <typename T>
  static sequence<T> merge(const sequence<T>& A, const sequence<T>& B) {
    auto R = sequence<T>::uninitialized(A.size() + B.size());
    merge_into(A.data(), A.size(), B.data(), B.size(), R.data(), std::less<>());
    return R;
  }

  /*-------------------------------------------------------------------*/
  /* A simple concurrent hash-based map (parlaylib's examples/hash_map.h)
   * Supports concurrent linearizable, insert, find and remove.
   * size(), keys() don't linearize with updates.
   * Requires the capacity to be  specified on construction.
   * No more than the capacity distinct keys can ever be added.
   * Once a key is added, removing it will empty the value and mark
   * the key as deleted, but only a value with the same key can use the
   * same slot (i.e. it still counts towards the capacity).
   * It uses locks, but holds them very briefly. */

  template <typename K,
            typename V,
            typename Hash = hash<K>,
            typename Equal = std::equal_to<> >
  struct hash_map {
   private:
    using index = unsigned long;
    long m;
    index first_index(K k) { return hash(k) % m;}
    index next_index(index h) { return (h + 1 == m) ? 0 : h + 1; }
    Hash hash;
    Equal equal;

    enum state : char {empty, full, locked, tomb};
    struct entry {
      std::atomic<state> status;
      K key;
      V value;
      entry() : status(empty) {}
    };
    sequence<entry> H;

   public:
    hash_map(long size, Hash&& hash = {}, Equal&& equal = {})
      : m(100 + static_cast<index>(1.5 * size)),
        hash(hash), equal(equal),
        H(sequence<entry>(m)) {}

    bool insert(const K& k, const V& v) {
      index i = first_index(k);
      long count = 0;
      while (true) {
        state status = H[i].status;
        if (status == full || status == tomb) {
          if (H[i].key == k) {
            if (status == full) return false;
            else if (H[i].status.compare_exchange_strong(status, locked)) {
              H[i].value = v;
              H[i].status = full;
              return true;
            }
          } else {
            if (count++ == std::min(1000l,m)) {
              std::cout << "Hash table overfull" << std::endl;
              return false;
            }
            i = next_index(i);
          }
        } else if (status == empty) {
          if (H[i].status.compare_exchange_strong(status, locked)) {
            H[i].key = k;
            H[i].value = v;
            H[i].status = full;
            return true;
          }
        }
      }
    }

    std::optional<V> find(const K& k) {
      index i = first_index(k);
      while (true) {
        if (H[i].status != full) return {};
        if (H[i].key == k) return H[i].value;
        i = next_index(i);
      }
    }

    std::optional<V> remove(const K& k) {
      index i = first_index(k);
      while (true) {
        if (H[i].status == empty || H[i].status == locked) return {};
        if (H[i].key == k) {
          state old = full;
          if (H[i].status == full &&
              H[i].status.compare_exchange_strong(old, tomb))
            return std::move(H[i].value);
          else return {};
        }
        i = next_index(i);
      }
    }

    // keys of the full slots, in slot order (parlay uses map_maybe)
    sequence<K> keys() {
      size_t n = H.size();
      size_t l = num_blocks(n, block_size);
      auto sums = sequence<size_t>::uninitialized(l);
      sliced_for(n, block_size, [&] (size_t i, size_t s, size_t e) {
        size_t r = 0;
        for (size_t j = s; j < e; j++) r += (H[j].status == full);
        sums[i] = r;
      });
      size_t total = scan_inplace(sums.data(), l);
      auto out = sequence<K>::uninitialized(total);
      sliced_for(n, block_size, [&] (size_t i, size_t s, size_t e) {
        size_t k = sums[i];
        for (size_t j = s; j < e; j++) {
          if (H[j].status == full) out[k++] = H[j].key;
        }
      });
      return out;
    }

    size_t size() {
      return keys().size();
    }
  };

  /*-------------------------------------------------------------------*/
  /* The main body */

  using Points = sequence<point>;

  struct triangle {
    tri t;
    sequence<point> conflicts;
    triangle(tri t, sequence<point> c) : t(t), conflicts(c) {}
  };

  struct Delaunay {
    using triangle_ptr = std::shared_ptr<triangle>;
    hash_map<tri,bool> mesh;
    hash_map<edge,triangle_ptr> edges;
    Points points;
    point_id n;

    point_id earliest(triangle_ptr& t) {
      return (t->conflicts.size() == 0) ? n : t->conflicts[0].id;}

    // filter points in the circumcicle of t1 and t2 which are in the
    // circumcicle of t
    auto filter_points(triangle_ptr& t1, triangle_ptr& t2, tri t) {
      auto a = merge(t1->conflicts, t2->conflicts);
      auto is_in_circle = in_circle(points[t[0]],points[t[1]],points[t[2]]);
      auto keep = tabulate(a.size(), [&] (long i) {
        return ((i != 0) && (a[i].id != a[i-1].id) &&
                ((i+1 < a.size() && a[i].id == a[i+1].id) ||
                 is_in_circle(a[i])));},500);
      return pack(a, keep);
    }

    // recursive routine
    // almost direct transcription from paper but limited to
    // 2d Delaunay instead of arbitrary convex hull
    void process_edge(triangle_ptr& t1, edge e, triangle_ptr& t2) {
      if (t1->conflicts.size() == 0 && t2->conflicts.size() == 0) {
        mesh.insert(t1->t,true); mesh.insert(t2->t,true);
        t1 = t2 = nullptr;
      } else if (earliest(t2) == earliest(t1)) {
        t1 = t2 = nullptr;
      } else {
        if (earliest(t2) < earliest(t1)) {
          std::swap(t2, t1); std::swap(e[0], e[1]);}
        point_id p = earliest(t1);
        tri t{e[0], e[1], p};
        t1 = std::make_shared<triangle>(t, filter_points(t1, t2, t));
        auto check_edge = [&] (edge e, triangle_ptr& tp) {
          auto key = (e[0] < e[1]) ? e : edge{e[1], e[0]};
          if (edges.insert(key, tp)) return;
          auto tt = *edges.remove(key);
          process_edge(tp, e, tt);};
        auto ta1 = t1; auto tb1 = t1;
        par_do3([&] {check_edge(edge{p, e[0]}, ta1);},
                [&] {check_edge(edge{e[1], p}, tb1);},
                [&] {process_edge(t1, e, t2);});
      }
    }

    // toplevel routine
    // The result is left in the hashmap mesh.  It includes all
    // triangles in the final mesh.
    // Assumes points are inside unit square to fit enclosing triangle.
    // And need to be in randomized order for good efficiency.
    Delaunay(const Points& P) :
        mesh(hash_map<tri,bool>(2*P.size())),
        edges(hash_map<edge,triangle_ptr>(6*P.size())),
        n(P.size()) {
      points = P;
      // enclosing triangle
      point p0{n,0.0,100.0};
      point p1{n+1,100.0,-100.0};
      point p2{n+2,-100.0,-100.0};
      points = append(P, Points({p0, p1, p2}));
      auto t = std::make_shared<triangle>(tri{n, n+1, n+2}, P);
      n += 3;
      auto te = std::make_shared<triangle>(tri{-1, -1, -1}, Points());
      std::shared_ptr<triangle> te2, te3, t2, t3;
      t2 = t3 = t;
      te2 = te3 = te;
      par_do3([&] {process_edge(t2, edge{p0.id,p1.id}, te2);},
              [&] {process_edge(t3, edge{p1.id,p2.id}, te3);},
              [&] {process_edge(t, edge{p2.id,p0.id}, te);});
    }
  };

  static sequence<tri> delaunay(const Points& P) {
    Delaunay dt(P);
    return dt.mesh.keys();
  }

};

} // end namespace
