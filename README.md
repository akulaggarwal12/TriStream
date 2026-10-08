# TriStream

Exact continuous subgraph matching over a pure edge stream.

Given an initial data graph G₀, an ordered stream of edge insertions and deletions, and a connected query graph Q,
TriStream computes, for every single update, the exact number of embeddings of Q that the update creates (ΔM⁺) or
destroys (ΔM⁻). The graph is never rescanned.

## Method in brief

TriStream turns each effective update into anchored tasks, one per query edge that can map onto the updated edge.
It matches them with **TriMatch**, a three-state matching core:

- **S2 (derivative signatures):** per-vertex counters of the one-hop and two-hop features that Q requires. Tasks, and
  candidates inside a search, that provably cannot be matched are dismissed.
- **S1 (cost model):** predicts each task's work and keeps the default matching order unless another is clearly
  cheaper. A deviation runs as a budgeted bet and is re-run with the default order if it overruns.
- **S3 (bounded search):** splits predicted-heavy tasks into disjoint pieces and searches them in decreasing predicted
  cost.

Only S2 removes work, and only provably empty work. S1 and S3 change the order and partition of the search, never its result. A controller keeps S2's maintenance only while its measured saving exceeds its cost.

Updates are processed in micro-batches on a versioned graph. Every update is still matched on its own snapshot:
an insertion on G_t, a deletion on G_{t-1}. So batching changes the execution, not the result.

TriMatch has two parallel executions. They run the same states, snapshots and control, and report identical counts:

| Execution   | Platform | Driver              | Binary                   |
| ----------- | -------- | ------------------- | ------------------------ |
| **TriSnap** | CPU      | `exec/CPU/main.cpp` | `exec/CPU/tristream_cpu` |
| **TriWarp** | GPU      | `exec/GPU/main.cu`  | `exec/GPU/tristream`     |

## Repository layout

```
exec/CPU/      main.cpp         TriSnap driver
               build.sh         builds tristream_cpu
               tristream_cpu    prebuilt binary (see Requirements)
exec/GPU/      main.cu          TriWarp driver
               build.sh         builds tristream
               tristream        prebuilt binary (see Requirements)
src/CPU/       host sources used by TriSnap: loading, CSR, query, matching plans, signatures
src/GPU/       the same host sources, plus the CUDA sources: versioned graph, signature store, matcher
include/csm/   shared headers
```

## Requirements

- **CPU build:** x86-64 Linux, g++ with C++17 and OpenMP (libstdc++ parallel mode is used for large sorts).
- **GPU build:** CUDA toolkit with `nvcc`, an NVIDIA GPU, and g++ with OpenMP as the host compiler.
- **Tested on:** 2× AMD EPYC 7452 (2 × 32 cores, 2 NUMA nodes), NVIDIA RTX A5000 (compute capability 8.6), Ubuntu
  20.04, g++ 9.4.0, CUDA 12.3, driver 550.
- The prebuilt binaries target that machine: the CPU binary uses `-march=native` and the GPU binary `sm_86`. On
  other hardware, rebuild.

## Build

```bash
cd exec/CPU && bash build.sh        # -> exec/CPU/tristream_cpu
cd exec/GPU && bash build.sh        # -> exec/GPU/tristream
```

- `nvcc` must be on the `PATH`, for example `export PATH=/usr/local/cuda/bin:$PATH`.
- For a GPU other than compute capability 8.6, set the architecture: `ARCH=sm_80 bash build.sh`.

## Run

```bash
exec/CPU/tristream_cpu --graph G --stream S --query Q [--directed] [--batch N] [--threads T] [--time-limit SEC]
                       [--name NAME] [--verbose]
exec/GPU/tristream     --graph G --stream S --query Q [--directed] [--batch N] [--time-limit SEC]
                       [--name NAME] [--verbose] [--device N] [--default-label L]
exec/GPU/tristream     --gpu-info
```

| Option                           | Meaning                                                                                |
| -------------------------------- | -------------------------------------------------------------------------------------- |
| `--graph`, `--stream`, `--query` | initial data graph, update stream, query graph (formats below)                         |
| `--directed`                     | treat the data graph and the query as directed (default: undirected)                   |
| `--batch N`                      | micro-batch size b (default 65536); a long stream starts with b/16 and doubles up to b |
| `--threads T`                    | CPU threads (default: all OpenMP threads)                                              |
| `--time-limit SEC`               | stop stream processing after SEC seconds; the reported counts are then lower bounds    |
| `--name NAME`                    | dataset name in the report (default: the graph file's folder name)                     |
| `--verbose`                      | detailed report per state, plus one machine-readable `RESULT key=value ...` line       |
| `--device N`                     | GPU index (GPU binary)                                                                 |
| `--default-label L`              | label for vertices without a `v` line (GPU binary; see below)                          |
| `--gpu-info`                     | print the GPU's properties and exit                                                    |

For one-socket CPU runs on a NUMA machine, bind threads and memory, for example
`numactl --cpunodebind=0 --membind=0 exec/CPU/tristream_cpu --threads 32 ...`.

### Example

```
$ exec/GPU/tristream --graph email10/initial --stream email10/s --query email10/triangle --name email10
==========================================
TriStream (TriWarp, GPU)
==========================================
Dataset        : email10
Nodes          : 36692
Edges          : 183831 (undirected)
Query          : triangle
Query details  : 3 vertices | 3 edges | max degree 2 | diameter 1 | 3 labels
Stream         : s
Stream updates : 8095 (2012 insertions, 6083 deletions)
Batches to GPU : 1 (batch size 65536; 509 of 8095 updates can touch the query)
Matches found  : 227 added, 435 deleted
Time (GPU)     : 2.28 ms
Speed          : 3.5433e+06 edges/sec
End-to-end time: 197.77 ms
==========================================
```

On the same input, `tristream_cpu --threads 32` reports the same `Matches found` line (227 added, 435 deleted).

- **Matches found:** the totals of the per-update ΔM⁺ and ΔM⁻ over the whole stream.
- **Time (CPU / GPU):** stream processing, from the relevance filter of the stream (plus the stream copy to the GPU)
  through the last micro-batch. Loading and preprocessing are excluded.
- **Speed:** updates of the original stream per second of that time.
- **End-to-end time:** the whole run, including parsing the input files and building the graph.

## Input formats

All three inputs are plain text with one record per line. Empty lines and lines starting with `#` or `%` are
ignored. Vertex ids and labels are non-negative integers. Edge labels are optional; they are matched only if the
query has edge labels.

**Initial graph (`--graph`)**

```
t <n> <m>             optional size hint
v <id> <label>        vertex label
e <u> <v> [label]     edge
```

**Update stream (`--stream`)**: one update per line, in stream order.

```
e <u> <v> [label]     insert edge (u, v); "+e" is also accepted
-e <u> <v> [label]    delete edge (u, v)
e -<u+1> -<v+1>       delete edge (u, v), alternative syntax (both ids shifted by one and negated)
v <id> <label>        label of a vertex first seen in the stream; "+v" is also accepted
-v <id>               ignored (deleting an isolated vertex cannot change any match)
```

- Self-loops are dropped.
- A duplicate insertion, or the deletion of an absent edge, is a no-op.
- Updates whose endpoint labels match no query edge cannot change any match. They are filtered out inside the timed
  region.

**Query (`--query`)**: `v <id> <label>` and `e <u> <v> [label]` lines. The query must be connected, with at most 32
vertices and 64 edges.

**Default vertex label.** A vertex without a `v` line gets the first of these that applies:

1. `--default-label` (GPU binary only);
2. the data's only label;
3. if the data is unlabelled, the query's only label;
4. 0.
