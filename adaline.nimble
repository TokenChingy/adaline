# Package
version       = "0.1.0"
author        = "Adaline"
description   = "Sparse Distributed Representations using Sparse Fingerprints and LSH"
license       = "MIT"
srcDir        = "."
bin           = @["adaline"]

# Dependencies
requires "nim >= 2.0.0"
requires "nimpy >= 0.2.0"

# Tasks
task release, "Compile adaline CLI in release mode":
  exec "nim c -d:release -o:adaline adaline.nim"

task python, "Build Python bindings":
  let nimpyPath = gorgeEx("nimble path nimpy").output.strip()
  exec "nim c -d:release --app:lib --path:\"" & nimpyPath & "\" -o:bindings/adaline.so bindings/adaline.nim"

task benchmark, "Compile benchmarks in release mode":
  exec "nim c -d:release -o:benchmarks/beir benchmarks/beir.nim"
  exec "nim c -d:release -o:benchmarks/longmemeval benchmarks/longmemeval.nim"
  exec "nim c -d:release -o:benchmarks/vision benchmarks/vision.nim"

task test, "Compile and run all tests":
  for f in @[
    "tests/domain/test_fingerprint.nim",
    "tests/domain/test_sdr_encoder.nim",
    "tests/domain/test_corpus_index.nim",
    "tests/domain/test_lexical_index.nim",
    "tests/domain/test_fingerprint_lsh.nim",
    "tests/domain/test_reranker.nim",
    "tests/domain/test_rrf_merger.nim",
    "tests/domain/test_weighted_jaccard.nim",
    "tests/domain/test_chunker.nim",
    "tests/domain/test_chunk.nim",
    "tests/domain/test_config.nim",
    "tests/domain/test_memory.nim",
    "tests/use_cases/test_insert_memory.nim",
    "tests/use_cases/test_search_memories.nim",
    "tests/use_cases/test_update_memory.nim",
    "tests/use_cases/test_delete_memory.nim"
  ]:
    exec "nim c -r " & f
