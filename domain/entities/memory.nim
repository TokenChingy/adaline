# Memory entity.
# Represents a stored memory with ID, content, relevance score,
# and timestamp.


type
  Memory* = object
    id*: uint64
    content*: string
    score*: float
    createdAt*: uint64
