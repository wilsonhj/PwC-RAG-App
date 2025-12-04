"""
Pytest configuration and shared fixtures.
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.rag import clear_all_documents, index_documents, DOCUMENTS


@pytest.fixture(scope="session")
def anyio_backend():
    """Use asyncio for async tests."""
    return "asyncio"


@pytest_asyncio.fixture
async def client():
    """Async HTTP client for API tests."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def clean_db():
    """
    Fixture to ensure clean database state.
    Clears all documents before and after test.
    """
    clear_all_documents()
    index_documents(DOCUMENTS)  # Re-index default docs
    yield
    # Cleanup after test
    clear_all_documents()
    index_documents(DOCUMENTS)
