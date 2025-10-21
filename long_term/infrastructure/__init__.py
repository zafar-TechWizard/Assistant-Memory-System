"""
Layer 2 Infrastructure Components

This module contains the foundational infrastructure components for Layer 2:
- Neo4j database client with connection pooling and retry logic
- Database configuration and connection management
- Performance monitoring and optimization tools
"""

from .neo4j_client import Neo4jClient

__all__ = ["Neo4jClient"]
