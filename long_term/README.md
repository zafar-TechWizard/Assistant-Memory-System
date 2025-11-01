# Long-Term Memory

This directory, `long_term`, is the core of the system's long-term memory. It's responsible for storing, managing, and retrieving information that needs to be persisted across conversations.

### Core Components:

*   **`memory_retrieval_engine.py`**: This is the main entry point for interacting with the long-term memory. It provides methods to store, search, and retrieve memories.
*   **`infrastructure/`**: This sub-directory contains the modules that handle the actual storage of the memories.
    *   **`neo4j_client.py`**: Manages the connection to the Neo4j graph database, where the relationships between different pieces of information are stored.
    *   **`redisHandler.py`**: Manages the connection to the Redis database, where the memories are stored.
*   **`models/`**: This sub-directory defines the data models for the memories.
    *   **`node_models.py`**: Defines the structure of the nodes in the graph database.
    *   **`relationship_models.py`**: Defines the structure of the relationships between the nodes in the graph database.

### Usage:

To use the long-term memory, you need to interact with the `MemoryRetrievalEngine` class in `memory_retrieval_engine.py`.

**Example: Storing a memory**

```python
from long_term.memory_retrieval_engine import MemoryRetrievalEngine

# Initialize the engine
retrieval_engine = MemoryRetrievalEngine()

# Store a memory
retrieval_engine.store_memory("The user's name is John Doe.")
```

**Example: Retrieving a memory**

```python
from long_term.memory_retrieval_engine import MemoryRetrievalEngine

# Initialize the engine
retrieval_engine = MemoryRetrievalEngine()

# Retrieve a memory
retrieved_memories = retrieval_engine.retrieve_memory("What is the user's name?")

# Print the retrieved memories
for memory in retrieved_memories:
    print(memory)
```
