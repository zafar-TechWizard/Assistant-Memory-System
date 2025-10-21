"""
Configuration for SOFI Memory System

This module provides configuration management for the memory system,
including database connections, performance settings, and feature flags.
"""

import os
from typing import Optional, Dict, Any
from pydantic import BaseSettings, Field


class MemoryConfig(BaseSettings):
    """Configuration settings for the SOFI Memory System"""
    
    # Database Configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    neo4j_username: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="password", description="Neo4j password")
    neo4j_database: str = Field(default="neo4j", description="Neo4j database name")
    
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_password: Optional[str] = Field(default=None, description="Redis password")
    redis_db: int = Field(default=0, description="Redis database number")
    
    chromadb_host: str = Field(default="localhost", description="ChromaDB host")
    chromadb_port: int = Field(default=8000, description="ChromaDB port")
    
    # Performance Settings
    max_connection_pool_size: int = Field(default=100, description="Maximum connection pool size")
    connection_timeout: int = Field(default=60, description="Connection timeout in seconds")
    query_timeout: int = Field(default=30, description="Query timeout in seconds")
    
    # Layer 1 Performance Targets
    layer1_response_time_target: float = Field(default=0.1, description="Layer 1 response time target in seconds")
    cache_hit_rate_target: float = Field(default=0.8, description="Target cache hit rate")
    max_concurrent_conversations: int = Field(default=100, description="Maximum concurrent conversations")
    
    # Layer 2 Performance Targets
    layer2_single_hop_target: float = Field(default=0.05, description="Layer 2 single-hop query target in seconds")
    layer2_multi_hop_target: float = Field(default=0.2, description="Layer 2 multi-hop query target in seconds")
    max_entities: int = Field(default=1000000, description="Maximum entities in knowledge graph")
    max_relationships: int = Field(default=10000000, description="Maximum relationships in knowledge graph")
    
    # Memory System Settings
    entity_extraction_confidence_threshold: float = Field(default=0.7, description="Minimum confidence for entity extraction")
    relationship_inference_confidence_threshold: float = Field(default=0.6, description="Minimum confidence for relationship inference")
    memory_consolidation_interval: int = Field(default=3600, description="Memory consolidation interval in seconds")
    
    # Feature Flags
    enable_entity_extraction: bool = Field(default=True, description="Enable entity extraction")
    enable_relationship_inference: bool = Field(default=True, description="Enable relationship inference")
    enable_memory_consolidation: bool = Field(default=True, description="Enable memory consolidation")
    enable_proactive_interactions: bool = Field(default=True, description="Enable proactive interactions")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    enable_performance_logging: bool = Field(default=True, description="Enable performance logging")
    
    class Config:
        env_prefix = "SOFI_MEMORY_"
        case_sensitive = False
        env_file = ".env"
    
    def get_neo4j_config(self) -> Dict[str, Any]:
        """Get Neo4j configuration dictionary"""
        return {
            "uri": self.neo4j_uri,
            "username": self.neo4j_username,
            "password": self.neo4j_password,
            "database": self.neo4j_database,
            "max_connection_pool_size": self.max_connection_pool_size,
            "connection_timeout": self.connection_timeout
        }
    
    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration dictionary"""
        return {
            "host": self.redis_host,
            "port": self.redis_port,
            "password": self.redis_password,
            "db": self.redis_db
        }
    
    def get_chromadb_config(self) -> Dict[str, Any]:
        """Get ChromaDB configuration dictionary"""
        return {
            "host": self.chromadb_host,
            "port": self.chromadb_port
        }


# Global configuration instance
config = MemoryConfig()


def get_config() -> MemoryConfig:
    """Get the global memory system configuration"""
    return config


def update_config(**kwargs) -> None:
    """Update configuration settings"""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration key: {key}")
