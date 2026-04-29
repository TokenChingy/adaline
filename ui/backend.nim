## Adaline WebView UI backend.
## Loads the React frontend and bridges Nim domain services via JSON-RPC-like
## messages over window.external.invoke / webview.eval.

import std/[os, json, tables]
import webview
import ../domain/services/memory/types
import ../domain/services/memory/init
import ../domain/services/memory/checkpoint
import ../domain/entities/config
import ../domain/entities/memory
import ../use_cases/insert_memory
import ../use_cases/search_memories
import ../use_cases/update_memory
import ../use_cases/delete_memory

type
  UiBackend = object
    service: MemoryService
    w: Webview

proc sendResponse(w: Webview; id: int; data: JsonNode) =
  let js = "window.dispatchResponse(" & $id & ", " & $data & ")"
  discard w.eval(js.cstring)

proc sendError(w: Webview; id: int; msg: string) =
  let js = "window.dispatchResponse(" & $id & ", {error:" & escapeJson(msg) & "})"
  discard w.eval(js.cstring)

proc handleInsert(be: var UiBackend; id: int; args: JsonNode) =
  let text = args{"text"}.getStr("")
  if text.len == 0:
    be.w.sendError(id, "empty text")
    return
  let output = insertMemory(be.service, InsertMemoryInput(content: text))
  be.w.sendResponse(id, %*{ "id": output.memoryId })

proc handleSearch(be: var UiBackend; id: int; args: JsonNode) =
  let query = args{"query"}.getStr("")
  let topK = args{"topK"}.getInt(10)
  if query.len == 0:
    be.w.sendResponse(id, newJArray())
    return
  let output = searchMemories(be.service, SearchMemoriesInput(query: query, topK: topK))
  var arr = newJArray()
  for mem in output.memories:
    arr.add(%*{
      "id": mem.id,
      "text": mem.content,
      "score": mem.score,
      "createdAt": mem.createdAt
    })
  be.w.sendResponse(id, arr)

proc handleDelete(be: var UiBackend; id: int; args: JsonNode) =
  let mid = args{"id"}.getBiggestInt(0).uint64
  discard deleteMemory(be.service, DeleteMemoryInput(memoryId: mid))
  be.w.sendResponse(id, %*{ "ok": true })

proc handleUpdate(be: var UiBackend; id: int; args: JsonNode) =
  let mid = args{"id"}.getBiggestInt(0).uint64
  let text = args{"text"}.getStr("")
  if text.len == 0:
    be.w.sendError(id, "empty text")
    return
  discard updateMemory(be.service, UpdateMemoryInput(memoryId: mid, content: text))
  be.w.sendResponse(id, %*{ "ok": true })

proc handleStats(be: var UiBackend; id: int) =
  let cfg = be.service.cfg
  let data = %*{
    "memoryCount": be.service.textCache.len,
    "corpusMemoryCount": be.service.corpus.numMemories,
    "fingerprintCount": be.service.storage.recordCount,
    "chunkCount": be.service.chunkToParent.len,
    "lexicalTermCount": be.service.lexical.postings.len,
    "fingerprintBits": cfg.fingerprintBits,
    "lshBands": cfg.lshBands,
    "lshRows": cfg.lshRows,
    "rrfK": cfg.rrfK,
    "dirichletMu": cfg.dirichletMu
  }
  be.w.sendResponse(id, data)

proc handleCheckpoint(be: var UiBackend; id: int) =
  checkpoint(be.service)
  be.w.sendResponse(id, %*{ "ok": true })

proc onExternalInvoke(be: var UiBackend; arg: string) =
  var req: JsonNode
  try:
    req = parseJson(arg)
  except:
    echo "Invalid JSON from UI: ", arg
    return
  let id = req{"id"}.getInt(-1)
  let methodName = req{"method"}.getStr("")
  let args = req{"args"}
  case methodName
  of "insert": handleInsert(be, id, args)
  of "search": handleSearch(be, id, args)
  of "delete": handleDelete(be, id, args)
  of "update": handleUpdate(be, id, args)
  of "stats": handleStats(be, id)
  of "checkpoint": handleCheckpoint(be, id)
  else:
    echo "Unknown method: ", methodName
    if id >= 0:
      be.w.sendError(id, "unknown method: " & methodName)

proc runUi*(dataDir: string) =
  ## Launch the Adaline desktop UI.
  let cfg = defaultEngineConfig()
  var be = UiBackend(
    service: initMemoryService(dataDir, cfg)
  )

  let distDir = getAppDir() / "ui" / "frontend" / "dist"
  let indexHtml = distDir / "index.html"
  let url = "file://" & indexHtml

  be.w = newWebView(title="Adaline", url=url, width=1200, height=800, resizable=true, debug=false)
  if be.w == nil:
    stderr.writeLine("Failed to create webview")
    quit(1)

  be.w.externalInvokeCB = proc (w: Webview; arg: string) =
    onExternalInvoke(be, arg)

  be.w.run()
