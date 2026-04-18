# Package
version       = "0.1.0"
author        = "Adaline"
description   = "Sparse Distributed Representations using Sparse Fingerprints and HNSW"
license       = "MIT"
srcDir        = "."
bin           = @["adaline"]

# Dependencies
requires "nim >= 2.0.0"

# Tasks
task release, "Compile adaline CLI in release mode":
  exec "nim c -d:release -o:adaline adaline.nim"

task benchmark, "Compile benchmarks in release mode":
  exec "nim c -d:release -o:benchmarks/benchmark_beir benchmarks/benchmark_beir.nim"
  exec "nim c -d:release -o:benchmarks/benchmark_longmemeval benchmarks/benchmark_longmemeval.nim"

task test, "Compile and run all tests":
  for f in @[
    "tests/domain/test_fingerprint.nim",
    "tests/domain/test_sdr_encoder.nim",
    "tests/domain/test_corpus_index.nim",
    "tests/domain/test_hnsw_node.nim",
    "tests/domain/test_lexical_index.nim",
    "tests/domain/test_minhash_lsh.nim",
    "tests/domain/test_reranker.nim",
    "tests/domain/test_rrf_merger.nim",
    "tests/domain/test_weighted_jaccard.nim",
    "tests/domain/test_chunker.nim",
    "tests/domain/test_memory_service.nim",
    "tests/use_cases/test_insert_memory.nim",
    "tests/use_cases/test_search_memories.nim"
  ]:
    exec "nim c -r " & f
