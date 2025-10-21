import asyncio
from typing import List, Dict, Any
from sofi_memory.layer2_long_term.infrastructure.neo4j_client import Neo4jClient, create_neo4j_client
from sofi_memory.processing.embedding_utils import EmbeddingUtils

class RetrievalEngine:
    """
    Handles the real-time retrieval of memories from the Layer 2 graph.
    It uses vector similarity to find the most relevant "entry points"
    into the graph and then traverses relationships to gather full context.
    """
    
    def __init__(self, neo4j_client: Neo4jClient):
        self.client = neo4j_client
        # The name of the vector index in your Neo4j database.
        # You must create this index in Neo4j for this to work.
        self.vector_index_name = "memory_vector_index" 

    async def retrieve_memories(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Finds and returns the most relevant memories for a given query.

        Args:
            query_text (str): The user's query or the text to find memories for.
            top_k (int): The maximum number of primary memories to retrieve.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                                 contains the retrieved memory node, its similarity
                                 score, and a list of related nodes.
        """
        print(f"Retrieving top {top_k} memories for query: '{query_text}'")
        
        # 1. Generate a vector embedding for the user's query.
        query_vector = EmbeddingUtils.generate_embedding(query_text)
        
        # 2. Execute the "Find and Expand" Cypher query.
        # This query performs two critical steps:
        #   a. FIND (Vector Search): It calls the vector index to find the `top_k`
        #      nodes with the most similar `content_vector`.
        #   b. EXPAND (Graph Traversal): For each node found, it optionally
        #      traverses one level of relationships to find directly connected nodes,
        #      providing immediate context.
        cypher_query = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector) YIELD node, score
        WITH node, score
        // Optional match to find directly connected nodes for context
        OPTIONAL MATCH (node)-[r:MEMORY_RELATIONSHIP]-(related_node)
        RETURN 
            node, 
            score, 
            collect(DISTINCT related_node) as related_nodes
        ORDER BY score DESC
        """
        
        params = {
            "index_name": self.vector_index_name,
            "top_k": top_k,
            "query_vector": query_vector
        }
        
        try:
            results = await self.client.execute_query(cypher_query, params)
            print(f"Found {len(results)} relevant memory contexts.")
            return results
        except Exception as e:
            print(f"An error occurred during memory retrieval: {e}")
            print(f"Please ensure the vector index '{self.vector_index_name}' exists in your Neo4j database.")
            return []

# --- Example Usage ---
async def main():
    print("\n--- Testing the Retrieval Engine ---")
    
    # IMPORTANT: For this test to work, you must:
    # 1. Have a running Neo4j instance.
    # 2. Have run the `setup.py` script to connect.
    # 3. Manually create the vector index in Neo4j with the following command:
    #    CREATE VECTOR INDEX memory_vector_index IF NOT EXISTS
    #    FOR (n:ExperienceMemory | n:KnowledgeMemory | n:RelationshipMemory)
    #    ON (n.content_vector) 
    #    OPTIONS {indexConfig: {'vector.dimensions': 384, 'vector.similarity_function': 'cosine'}}
    # 4. Have at least one node with a 'content_vector' property in the database.
    
    neo4j_client = create_neo4j_client()
    await neo4j_client.connect()
    
    retriever = RetrievalEngine(neo4j_client)
    
    # Let's imagine we previously stored a memory about John helping with Python.
    # Now, the user asks a semantically similar question.
    user_query = "Who was it that helped me with that coding problem?"
    
    retrieved_data = await retriever.retrieve_memories(user_query)
    
    if not retrieved_data:
        print("\nNo memories retrieved. This is expected on an empty database.")
        print("To test properly, add a node with a 'content_vector' property.")
    else:
        print("\n--- Retrieved Memories ---")
        for item in retrieved_data:
            main_node = item['node']
            score = item['score']
            related = item['related_nodes']
            
            print(f"\n[+] Main Node Content: '{main_node.get('content')}'")
            print(f"    - Context: {main_node.get('memory_context')}")
            print(f"    - Relevance Score: {score:.4f}")
            if related:
                print("    - Related Context:")
                for rel_node in related:
                    print(f"      - '{rel_node.get('content')}' ({rel_node.get('memory_context')})")

    await neo4j_client.disconnect()

if __name__ == '__main__':
    asyncio.run(main())
