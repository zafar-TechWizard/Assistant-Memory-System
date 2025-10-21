"""
Setup script for SOFI Memory System

This script helps set up the memory system infrastructure including:
- Database connections and schema creation
- Initial configuration
- Health checks and validation
"""

import asyncio
import logging
from typing import Dict, Any

from .config import get_config
from .long_term.infrastructure.neo4j_client import create_neo4j_client


logger = logging.getLogger(__name__)


async def setup_neo4j_database() -> bool:
    """Set up Neo4j database with schema and constraints"""
    logger.info("Setting up Neo4j database...")
    
    config = get_config()
    client = create_neo4j_client(**config.get_neo4j_config())
    
    try:
        # Connect to database
        await client.connect()
        logger.info("✓ Connected to Neo4j database")
        
        # Create schema
        await client.create_constraints_and_indexes()
        logger.info("✓ Created database schema")
        
        # Test basic operations
        result = await client.execute_query("RETURN 'SOFI Memory System Ready' as status")
        logger.info(f"✓ Database test successful: {result[0]['status']}")
        
        # Get database info
        db_info = await client.get_database_info()
        logger.info(f"✓ Database info: {len(db_info.get('node_counts', []))} node types, {len(db_info.get('relationship_counts', []))} relationship types")
        
        await client.disconnect()
        return True
        
    except Exception as e:
        logger.error(f"✗ Neo4j setup failed: {e}")
        return False


async def validate_environment() -> Dict[str, bool]:
    """Validate that all required services are available"""
    logger.info("Validating environment...")
    
    results = {}
    
    # Test Neo4j connection
    try:
        config = get_config()
        client = create_neo4j_client(**config.get_neo4j_config())
        await client.connect()
        health = await client.health_check()
        results['neo4j'] = health['status'] == 'healthy'
        await client.disconnect()
    except Exception as e:
        logger.error(f"Neo4j validation failed: {e}")
        results['neo4j'] = False
    
    # Test Redis connection (placeholder - will implement when we add Redis)
    results['redis'] = True  # Placeholder
    
    # Test ChromaDB connection (placeholder - will implement when we add ChromaDB)
    results['chromadb'] = True  # Placeholder
    
    return results


async def run_health_checks() -> Dict[str, Any]:
    """Run comprehensive health checks on all components"""
    logger.info("Running health checks...")
    
    health_status = {}
    
    # Neo4j health check
    try:
        config = get_config()
        client = create_neo4j_client(**config.get_neo4j_config())
        await client.connect()
        health_status['neo4j'] = await client.health_check()
        await client.disconnect()
    except Exception as e:
        health_status['neo4j'] = {
            'status': 'unhealthy',
            'error': str(e)
        }
    
    return health_status


async def setup_memory_system() -> bool:
    """Complete setup of the memory system"""
    logger.info("Starting SOFI Memory System setup...")
    
    # Validate environment
    validation_results = await validate_environment()
    logger.info(f"Environment validation: {validation_results}")
    
    if not all(validation_results.values()):
        logger.error("Environment validation failed. Please check your database connections.")
        return False
    
    # Set up Neo4j database
    if not await setup_neo4j_database():
        logger.error("Neo4j database setup failed.")
        return False
    
    # Run health checks
    health_status = await run_health_checks()
    logger.info(f"Health check results: {health_status}")
    
    logger.info("🎉 SOFI Memory System setup completed successfully!")
    return True


async def main():
    """Main setup function"""
    logging.basicConfig(level=logging.INFO)
    
    success = await setup_memory_system()
    if success:
        print("✅ SOFI Memory System is ready!")
    else:
        print("❌ SOFI Memory System setup failed!")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
