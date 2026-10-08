#include "csm/text_scanner.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

namespace csm {

MappedFile::MappedFile(const std::string& path) {
  fd_ = ::open(path.c_str(), O_RDONLY);
  if (fd_ < 0) fatal("cannot open '%s': %s", path.c_str(), std::strerror(errno));
  struct stat st {};
  if (::fstat(fd_, &st) != 0) fatal("cannot stat '%s'", path.c_str());
  size_ = static_cast<size_t>(st.st_size);
  if (size_ == 0) {
    data_ = "";
    return;
  }
  void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (p == MAP_FAILED) fatal("cannot mmap '%s': %s", path.c_str(), std::strerror(errno));
  ::madvise(p, size_, MADV_SEQUENTIAL);
  data_ = static_cast<const char*>(p);
}

MappedFile::~MappedFile() {
  if (size_ > 0 && data_) ::munmap(const_cast<char*>(data_), size_);
  if (fd_ >= 0) ::close(fd_);
}

static inline bool is_space(char c) { return c == ' ' || c == '\t' || c == '\r'; }

bool LineScanner::next_line() {
  while (cur_ < end_) {
    const char* nl = static_cast<const char*>(std::memchr(cur_, '\n', static_cast<size_t>(end_ - cur_)));
    const char* le = nl ? nl : end_;
    const char* p = cur_;
    ++line_no_;
    cur_ = nl ? nl + 1 : end_;
    while (p < le && is_space(*p)) ++p;
    if (p == le || *p == '#' || *p == '%') continue;
    line_end_ = le;
    tok_start_ = p;
    return true;
  }
  return false;
}

bool LineScanner::next_token(std::string_view& tok) {
  const char* p = tok_start_;
  while (p < line_end_ && is_space(*p)) ++p;
  if (p >= line_end_) return false;
  const char* q = p;
  while (q < line_end_ && !is_space(*q)) ++q;
  tok = std::string_view(p, static_cast<size_t>(q - p));
  tok_start_ = q;
  return true;
}

bool parse_raw_id(std::string_view s, raw_id_t& out) {
  if (s.empty() || s.size() > 38) return false;
  unsigned __int128 x = 0;
  for (char c : s) {
    if (c < '0' || c > '9') return false;
    x = x * 10 + static_cast<unsigned>(c - '0');
  }
  out = raw_id_t{static_cast<uint64_t>(x >> 64), static_cast<uint64_t>(x)};
  return true;
}

bool parse_u64(std::string_view s, uint64_t& out) {
  if (s.empty() || s.size() > 19) return false;
  uint64_t x = 0;
  for (char c : s) {
    if (c < '0' || c > '9') return false;
    x = x * 10 + static_cast<unsigned>(c - '0');
  }
  out = x;
  return true;
}

}  // namespace csm
