import ./types
import ../../algorithms/fingerprint_lsh
import ../../algorithms/lexical_index
import ../../algorithms/corpus_index
import std/os

export types

proc checkpoint*(service: var MemoryService) =
  ## Persist in-memory indexes to disk so future restarts can skip WAL replay.
  let walOffset = service.storage.walSize
  let lshPath = service.storage.dataDir / "lsh.bin"
  let lexicalPath = service.storage.dataDir / "lexical.bin"
  let corpusPath = service.storage.dataDir / "corpus.bin"
  saveLsh(service.lsh, lshPath, walOffset)
  saveLexical(service.lexical, lexicalPath, walOffset)
  saveCorpus(service.corpus, corpusPath, walOffset)
