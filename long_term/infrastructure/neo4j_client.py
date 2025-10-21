"""
Neo4j Database Client for SOFI Memory System

This module provides a robust Neo4j client with connection pooling, retry logic,
and performance optimization for the SOFI memory system's Layer 2 knowledge graph.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from contextlib import asynccontextmanager
from datetime import datetime
import json

try:
    from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession, AsyncTransaction
    from neo4j.exceptions import ServiceUnavailable, TransientError, DatabaseError
except ImportError:
    raise ImportError(
        "Neo4j driver not installed. Please install with: pip install neo4j==5.15.0"
    )

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


logger = logging.getLogger(__name__)


class Neo4jConfig(BaseModel):
    """Configuration for Neo4j connection"""
    uri: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    username: str = Field(default="neo4j", description="Database username")
    password: str = Field(default="password", description="Database password")
    database: str = Field(default="neo4j", description="Database name")
    max_connection_lifetime: int = Field(default=3600, description="Max connection lifetime in seconds")
    max_connection_pool_size: int = Field(default=100, description="Max connection pool size")
    connection_acquisition_timeout: int = Field(default=60, description="Connection acquisition timeout in seconds")
    max_transaction_retry_time: int = Field(default=30, description="Max transaction retry time in seconds")


class Neo4jClient:
    """
    Robust Neo4j client with connection pooling, retry logic, and performance optimization.
    
    Features:
    - Async/await support for high performance
    - Connection pooling and lifecycle management
    - Automatic retry logic for transient failures
    - Transaction management with proper error handling
    - Performance monitoring and query optimization
    - Health checks and connection validation
    """
    
    def __init__(self, config: Neo4jConfig):
        self.config = config
        self.driver: Optional[AsyncDriver] = None
        self._connection_pool = None
        self._is_connected = False
        
    async def connect(self) -> None:
        """Initialize connection to Neo4j database"""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
                max_connection_lifetime=self.config.max_connection_lifetime,
                max_connection_pool_size=self.config.max_connection_pool_size,
                connection_acquisition_timeout=self.config.connection_acquisition_timeout,
                max_transaction_retry_time=self.config.max_transaction_retry_time
            )
            
            # Test connection
            await self._test_connection()
            self._is_connected = True
            logger.info(f"Successfully connected to Neo4j at {self.config.uri}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close connection to Neo4j database"""
        if self.driver:
            await self.driver.close()
            self._is_connected = False
            logger.info("Disconnected from Neo4j")
    
    async def _test_connection(self) -> None:
        """Test database connection"""
        async with self.driver.session(database=self.config.database) as session:
            result = await session.run("RETURN 1 as test")
            await result.single()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ServiceUnavailable, TransientError))
    )
    async def execute_query(
        self, 
        query: str, 
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query with retry logic and error handling.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            database: Database name (defaults to config database)
            
        Returns:
            List of result records as dictionaries
            
        Raises:
            DatabaseError: For database-related errors
            ServiceUnavailable: For connection issues
        """
        if not self._is_connected:
            raise ConnectionError("Not connected to Neo4j database")
        
        database = database or self.config.database
        parameters = parameters or {}
        
        try:
            async with self.driver.session(database=database) as session:
                result = await session.run(query, parameters)
                records = []
                async for record in result:
                    records.append(dict(record))
                return records
                
        except (ServiceUnavailable, TransientError) as e:
            logger.warning(f"Transient error executing query, retrying: {e}")
            raise
        except DatabaseError as e:
            logger.error(f"Database error executing query: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error executing query: {e}")
            raise
    
    @asynccontextmanager
    async def transaction(self, database: Optional[str] = None):
        """
        Context manager for database transactions with automatic rollback on error.
        
        Usage:
            async with client.transaction() as tx:
                await tx.run("CREATE (n:Person {name: $name})", {"name": "John"})
                await tx.run("CREATE (n:Person {name: $name})", {"name": "Jane"})
        """
        if not self._is_connected:
            raise ConnectionError("Not connected to Neo4j database")
        
        database = database or self.config.database
        session = self.driver.session(database=database)
        transaction = await session.begin_transaction()
        
        try:
            yield transaction
            await transaction.commit()
        except Exception as e:
            await transaction.rollback()
            logger.error(f"Transaction rolled back due to error: {e}")
            raise
        finally:
            await session.close()
    
    async def create_constraints_and_indexes(self) -> None:
        """Create necessary constraints and indexes for optimal performance"""
        constraints_and_indexes = [
            # Memory node constraints
            "CREATE CONSTRAINT experience_memory_id_unique IF NOT EXISTS FOR (e:ExperienceMemory) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT knowledge_memory_id_unique IF NOT EXISTS FOR (k:KnowledgeMemory) REQUIRE k.id IS UNIQUE", 
            "CREATE CONSTRAINT relationship_memory_id_unique IF NOT EXISTS FOR (r:RelationshipMemory) REQUIRE r.id IS UNIQUE",
            "CREATE CONSTRAINT current_memory_id_unique IF NOT EXISTS FOR (c:CurrentMemory) REQUIRE c.id IS UNIQUE",
            
            # Performance indexes for memory contexts
            "CREATE INDEX experience_timestamp_index IF NOT EXISTS FOR (e:ExperienceMemory) ON (e.timestamp)",
            "CREATE INDEX experience_participants_index IF NOT EXISTS FOR (e:ExperienceMemory) ON (e.participants)",
            "CREATE INDEX knowledge_concept_index IF NOT EXISTS FOR (k:KnowledgeMemory) ON (k.concept)",
            "CREATE INDEX knowledge_category_index IF NOT EXISTS FOR (k:KnowledgeMemory) ON (k.category)",
            "CREATE INDEX relationship_person_index IF NOT EXISTS FOR (r:RelationshipMemory) ON (r.person_name)",
            "CREATE INDEX relationship_type_index IF NOT EXISTS FOR (r:RelationshipMemory) ON (r.relationship_type)",
            "CREATE INDEX current_focus_index IF NOT EXISTS FOR (c:CurrentMemory) ON (c.current_focus)",
            "CREATE INDEX current_time_context_index IF NOT EXISTS FOR (c:CurrentMemory) ON (c.time_context)",
            
            # Cross-context relevance indexes
            "CREATE INDEX experience_knowledge_relevance_index IF NOT EXISTS FOR (e:ExperienceMemory) ON (e.knowledge_relevance)",
            "CREATE INDEX experience_relationship_relevance_index IF NOT EXISTS FOR (e:ExperienceMemory) ON (e.relationship_relevance)",
            "CREATE INDEX knowledge_experience_relevance_index IF NOT EXISTS FOR (k:KnowledgeMemory) ON (k.experience_relevance)",
            "CREATE INDEX relationship_experience_relevance_index IF NOT EXISTS FOR (r:RelationshipMemory) ON (r.experience_relevance)",
            
            # Memory relationship indexes
            "CREATE INDEX memory_relationship_strength_index IF NOT EXISTS FOR ()-[r:MEMORY_RELATIONSHIP]-() ON (r.strength)",
            "CREATE INDEX memory_relationship_confidence_index IF NOT EXISTS FOR ()-[r:MEMORY_RELATIONSHIP]-() ON (r.confidence)",
            "CREATE INDEX memory_relationship_type_index IF NOT EXISTS FOR ()-[r:MEMORY_RELATIONSHIP]-() ON (r.relationship_type)",
            "CREATE INDEX memory_relationship_created_index IF NOT EXISTS FOR ()-[r:MEMORY_RELATIONSHIP]-() ON (r.created_date)",
        ]
        
        for constraint_or_index in constraints_and_indexes:
            try:
                await self.execute_query(constraint_or_index)
                logger.info(f"Created constraint/index: {constraint_or_index.split()[2]}")
            except Exception as e:
                logger.warning(f"Failed to create constraint/index: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on database connection"""
        try:
            start_time = datetime.now()
            result = await self.execute_query("RETURN 1 as health_check")
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "database": self.config.database,
                "uri": self.config.uri,
                "connected": self._is_connected,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "database": self.config.database,
                "uri": self.config.uri,
                "connected": False,
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_database_info(self) -> Dict[str, Any]:
        """Get database information and statistics"""
        try:
            # Get node counts
            node_counts = await self.execute_query("""
                MATCH (n)
                RETURN labels(n) as labels, count(n) as count
                ORDER BY count DESC
            """)
            
            # Get relationship counts
            relationship_counts = await self.execute_query("""
                MATCH ()-[r]->()
                RETURN type(r) as type, count(r) as count
                ORDER BY count DESC
            """)
            
            # Get database size
            db_info = await self.execute_query("CALL db.info()")
            
            return {
                "node_counts": node_counts,
                "relationship_counts": relationship_counts,
                "database_info": db_info[0] if db_info else {},
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return {"error": str(e)}
    
    def __del__(self):
        """Cleanup on object destruction"""
        if self.driver and self._is_connected:
            asyncio.create_task(self.disconnect())


# Factory function for creating Neo4j client
def create_neo4j_client(
    uri: str = "bolt://localhost:7687",
    username: str = "neo4j", 
    password: str = "password",
    database: str = "neo4j",
    **kwargs
) -> Neo4jClient:
    """
    Factory function to create a Neo4j client with default configuration.
    
    Args:
        uri: Neo4j connection URI
        username: Database username
        password: Database password
        database: Database name
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured Neo4jClient instance
    """
    config = Neo4jConfig(
        uri=uri,
        username=username,
        password=password,
        database=database,
        **kwargs
    )
    return Neo4jClient(config)
