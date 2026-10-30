# Evaluation

A pre-registered test of whether the library helps on a practical problem.
The protocol, harness and raw records are kept in a separate repository;
this page summarises them. The results hold for the problems, budgets and
machine below.

## Setup

- **Problem:** economic dispatch with valve-point costs. A power demand is
  split across generating units whose cost curves have a ridge every few MW,
  so the total cost has many local minima. Four instances from Pedroso, Kubo
  and Viana and from CEC 2011: 13 units at two demands, 40 units, and 140
  units with 11 prohibited zones on 4 of them.
- **Grid:** outputs on a 0.1 MW grid, so a dispatch is a bounded composition
  and a prohibited zone is a gap. The exact grid optimum, by dynamic
  programming over the units, is the ground truth; results are gaps to it.
- **Runs:** about 150,000 evaluations per run (the CEC 2011 budget), 25 runs
  per method. Comparisons are two-sided Mann-Whitney U tests with Holm's
  adjustment, against criteria fixed before any run.
- **Speed:** valid points per second on an i9-11980HK (8 cores), threads
  pinned one per core, the minimum of 7 repeats, timed only on an idle
  machine.

A first test, post hoc ensembling of tabular models, was stopped early: no
method beat greedy selection, which is hard to beat on an objective that is
close to convex.

## Where the library helps

- **Exact uniform sampling.** The library's samplers are uniform over the
  valid compositions: the total variation from the exact distribution of a
  part is at most 0.0023 at 10^6 points, the noise level. Drawing unit by
  unit (each part uniform within what keeps the rest feasible) is always
  valid but far from uniform (0.03 to 0.95). Rejection is not practical:
  stars and bars accepts 10^-3.5, 10^-8.1, 10^-18.4 and 10^-53.0 of its
  draws on the four spaces.
- **Prohibited zones as gaps.** Counted in the table, the zones never need
  filtering. With that and the batched walks, uniform sampling of the
  140-unit space became 8.7 to 9.7 times faster (2.0 to 2.6 times on the
  spaces without zones).
- **Parallel sampling.** At 8 threads the library is 1.7 to 2.5 times as fast
  as unit-by-unit drawing in a process pool on the 13- and 40-unit spaces.
- **Sampling by region success.** 64 regions, with most of each round's
  points given to the regions that hold the cheapest points found so far,
  beat the same budget spread evenly by a rank sequence on three of the four
  problems (median gap 3.19% to 2.71%, 2.12% to 1.65%, 12.58% to 12.23%; no
  significant difference on 40 units).
- **Local search without repeats.** A prototype, not in the library: moves
  between pairs of units, never evaluating a composition twice. It beat the
  best of DE, PSO and CMA-ES on three of the four problems.

## Where it does not help

- **One-thread speed.** Unit-by-unit drawing is 1.5 to 5.6 times faster on
  one thread, at the cost of uniformity. The library's efficiency at 8
  threads is 41% to 62%, and on the 140-unit space the process pool is
  ahead at 8 threads too.
- **Spread sequences for random search.** Random search with the library's
  rank or Sobol sequences did no better than with independent draws (within
  0.2 percentage points of gap).
- **The unit-cube codec for optimisers.** Mixed: better for DE, PSO and
  CMA-ES on one 13-unit problem and for PSO on 140 units, worse for DE and
  CMA-ES on 40 and 140 units and for all three on the other 13-unit problem.
- **Regions combined with optimisers**, at a fixed total of evaluations with
  64 evaluated per round:
  - As a start: sampling by region success for a third of the budget, then
    local search or CMA-ES from the best point, was no better than the
    optimiser alone.
  - As parallel workers: one optimiser per region, the round's 64
    evaluations shared among the regions by success, never beat one
    optimiser over the whole space. Local search put about 87% of its
    evaluations into one region. With one worker per region and equal
    shares (an exploratory comparison), the loss was largest on the larger
    problems: on 140 units, 1.37% against 0.02% for local search.

Not tested: one optimiser per worker with a full budget of its own, the case
where evaluations are cheap and the workers would otherwise idle.
