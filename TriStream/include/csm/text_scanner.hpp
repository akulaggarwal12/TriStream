// Fast line/token scanning over a memory-mapped text file.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "csm/id_map.hpp"

namespace csm {

class MappedFile {
 public:
  explicit MappedFile(const std::string& path);
  ~MappedFile();
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  const char* begin() const { return data_; }
  const char* end() const { return data_ + size_; }
  size_t size() const { return size_; }

 private:
  const char* data_ = nullptr;
  size_t size_ = 0;
  int fd_ = -1;
};

class LineScanner {
 public:
  LineScanner(const char* begin, const char* end) : cur_(begin), end_(end) {}

  // Moves to the next line that is non-empty and not a comment ('#' or '%').
  bool next_line();
  // Next whitespace-separated token of the current line; false at end of line.
  bool next_token(std::string_view& tok);
  size_t line_no() const { return line_no_; }
  const char* line_begin() const { return tok_start_; }
  const char* line_end() const { return line_end_; }

 private:
  const char* cur_;
  const char* end_;
  const char* line_end_ = nullptr;
  const char* tok_start_ = nullptr;
  size_t line_no_ = 0;
};

// Decimal parsers; they reject empty strings, signs and non-digits.
bool parse_raw_id(std::string_view s, raw_id_t& out);
bool parse_u64(std::string_view s, uint64_t& out);

// Strips one leading '-' for deletion case. Returns true if stripped.
inline bool strip_minus(std::string_view& s) {
  if (!s.empty() && s.front() == '-') {
    s.remove_prefix(1);
    return true;
  }
  return false;
}

}  // namespace csm
