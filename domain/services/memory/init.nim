## Memory service initialisation.
## Creates or opens the memory-mapped storage (WAL, fingerprint store,
## chunks store) and replays WAL and chunk mappings.


import ./types
import ../../entities/config
import ../../entities/fingerprint
import ../../algorithms/fingerprint_lsh
import ../../algorithms/lexical_index
import ../../algorithms/corpus_index
import ../../algorithms/chunker
import ../../../infrastructure/mmapped_storage
import std/[tables, sets, strutils, random, os]

export types
export config

proc initMemoryService*(dataDir: string; cfg: EngineConfig = defaultEngineConfig()): MemoryService =
  randomize()
  result.storage = initStorage(dataDir)
  result.cfg = cfg
  result.lsh = initLshIndex(cfg)
  result.lexical = LexicalIndex(mu: cfg.dirichletMu)
  result.corpus = CorpusIndex()
  result.textCache = initTable[uint64, string]()
  result.lowerTextCache = initTable[uint64, string]()
  result.tokenCache = initTable[uint64, HashSet[string]]()
  result.timestampCache = initTable[uint64, uint64]()
  result.chunkToParent = initTable[uint64, uint64]()
  result.parentToChunks = initTable[uint64, seq[uint64]]()

  let lshPath = dataDir / "lsh.bin"
  let lexicalPath = dataDir / "lexical.bin"
  let corpusPath = dataDir / "corpus.bin"
  var lshOffset, lexicalOffset, corpusOffset: uint64
  var haveLsh = false
  var haveLexical = false
  var haveCorpus = false
  if fileExists(lshPath) and fileExists(lexicalPath) and fileExists(corpusPath):
    result.lsh = loadLsh(cfg, lshPath, lshOffset)
    result.lexical = loadLexical(lexicalPath, lexicalOffset)
    result.corpus = loadCorpus(corpusPath, corpusOffset)
    haveLsh = lshOffset > 0 or result.lsh.buckets.len > 0
    haveLexical = lexicalOffset > 0 or result.lexical.postings.len > 0
    haveCorpus = corpusOffset > 0 or result.corpus.memFreqs.len > 0

  let persistedOffset = if haveLsh and haveLexical and haveCorpus and
                          lshOffset == lexicalOffset and lexicalOffset == corpusOffset:
                          lshOffset
                        else:
                          0'u64

  let chunkEntries = replayChunks(result.storage)
  var maxId: uint64 = 0
  var hasData = false
  for (parentId, chunkId) in chunkEntries:
    result.chunkToParent[chunkId] = parentId
    result.parentToChunks.mgetOrPut(parentId, @[]).add(chunkId)
    if chunkId > maxId:
      maxId = chunkId
    hasData = true

  let entries = replayWal(result.storage)
  var walPos: uint64 = 0
  for (parentId, timestamp, text) in entries:
    result.textCache[parentId] = text
    result.lowerTextCache[parentId] = text.toLowerAscii()
    result.tokenCache[parentId] = initHashSet[string]()
    for token in text.toLowerAscii().split(AllChars - Letters - Digits):
      if token.len > 0:
        result.tokenCache[parentId].incl(token)
    result.timestampCache[parentId] = timestamp
    if parentId > maxId:
      maxId = parentId
    hasData = true

    let entrySize = uint64(sizeof(uint64) + sizeof(uint64) + sizeof(uint32) + text.len)
    let entryStart = walPos
    walPos += entrySize

    let storedChunkIds = result.parentToChunks.getOrDefault(parentId, @[])
    let chunkTexts = splitIntoChunks(text, cfg)
    let effectiveChunkIds = if storedChunkIds.len > 0:
                              storedChunkIds
                            else:
                              @[parentId]
    let numChunks = min(effectiveChunkIds.len, chunkTexts.len)

    if entryStart < persistedOffset:
      result.corpus.addMemory(text)
    else:
      result.corpus.addMemory(text)
      for i in 0 ..< numChunks:
        let chunkId = effectiveChunkIds[i]
        let chunkText = chunkTexts[i]
        addMemory(result.lexical, chunkId, chunkText)
        var fp: Fingerprint
        result.storage.readFingerprint(chunkId, addr fp)
        insertLsh(result.lsh, addr fp, chunkId)

  if hasData:
    result.storage.syncRecordCount(maxId + 1)
