// process_edge recurses too deeply for the default fiber stacks (64KB on
// arm64, 128KB on x64); must be defined before any taskparts header
#ifndef TASKPARTS_FIBER_STACK_SZB
#define TASKPARTS_FIBER_STACK_SZB (1 << 20)
#endif

#include <stdio.h>
#include <optional>
#include <random>

#include <taskparts/benchmark.hpp>
#include "delaunay_nativeforkjoin.hpp"

// Delaunay triangulation of n random points in the unit square, run on
// native fork-join fibers (port of parlaylib's examples/delaunay.cpp).
// Usage: delaunay_nativeforkjoin.<variant> -n <nb points>

int main() {
  using namespace delaunay_nativefj;
  using dt = fj<taskparts::bench_scheduler>;
  point_id n = taskparts::cmdline::parse_or_default_long("n", 1000000);
  // the points are created and freed inside the scheduler, because
  // sequences use parallel loops to copy and destroy their elements
  std::optional<dt::Points> points;
  size_t nb_triangles = 0;
  taskparts::benchmark_nativeforkjoin([&] (auto sched) {
    nb_triangles = dt::delaunay(*points).size();
  }, [&] (auto sched) { // setup
    // generate n random points in a unit square (same points as parlay's example)
    random_generator gen(0);
    std::uniform_real_distribution<real> dis(0.0,1.0);
    points.emplace(dt::tabulate(n, [&] (point_id i) -> point {
      auto r = gen[i];
      return point{i, dis(r), dis(r)};}));
  }, [&] (auto sched) { // teardown
    points.reset();
  });
  printf("number of triangles in the mesh = %lu\n", nb_triangles);
  return 0;
}
