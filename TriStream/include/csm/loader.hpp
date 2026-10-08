// Loads the three inputs of CSM
#pragma once

#include <string>

#include "csm/host_graph.hpp"
#include "csm/id_map.hpp"
#include "csm/query_graph.hpp"
#include "csm/update_stream.hpp"

namespace csm {

struct LoadOptions {
  std::string graph_path;
  std::string stream_path;
  std::string query_path;
  bool directed = false;
  bool has_default_vlabel = false;
  uint64_t default_vlabel = 0;
};

struct LoadTimes {
  double graph_read_ms = 0, stream_read_ms = 0, query_read_ms = 0, csr_build_ms = 0;
};

struct Dataset {
  HostGraph graph;
  UpdateStream stream;
  QueryGraph query;
  IdMap ids;
  LabelMap vlabels{false};
  LabelMap elabels{true};
  LoadTimes times;
  label_t default_vlabel = 0;
  GraphReadStats graph_read;
  // relevance pre-filter results
  bool relevance_filtered = false;
  uint64_t rel_edges_dropped = 0;       // initial edges that can match no query edge
  uint64_t stream_orig_size = 0;        // number of updates in the original stream
  double rel_filter_ms = 0.0;             // initial-edge filter (and the stream filter unless deferred)
  bool stream_filter_pending = false;     // defer_stream_filter: relevance_filter_stream() still has to run
  double stream_filter_ms = 0.0;          // time of the stream filter (inside the timed region when deferred)
  std::vector<Update> stream_raw;
};

Dataset load_dataset(const LoadOptions& opt);

// The stream part of the relevance pre-filter
void relevance_filter_stream(Dataset& d, const label_t* vlabel, bool directed);

void print_dataset_report(const Dataset& d, FILE* out);

}  // namespace csm
