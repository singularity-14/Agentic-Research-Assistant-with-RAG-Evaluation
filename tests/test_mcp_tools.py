"""Tests for MCP server tools."""

from __future__ import annotations

import pytest


class TestMcpAuth:
    def test_verify_valid_token(self):
        from src.mcp_server.auth import verify_token
        from src.config import settings

        assert verify_token(settings.mcp_auth_token) is True

    def test_verify_invalid_token(self):
        from src.mcp_server.auth import verify_token

        assert verify_token("wrong-token-xyz") is False

    def test_verify_empty_token(self):
        from src.mcp_server.auth import verify_token

        assert verify_token("") is False


class TestMcpServerRegistration:
    def test_mcp_server_has_tools(self):
        """Verify the MCP server registers the expected tools."""
        from src.mcp_server.server import mcp

        # FastMCP exposes registered tools via _tool_manager
        tool_names = list(mcp._tool_manager._tools.keys())
        assert "search_cern_docs" in tool_names
        assert "get_paper_summary" in tool_names
        assert "list_indexed_categories" in tool_names

    def test_mcp_server_has_resources(self):
        from src.mcp_server.server import mcp

        resource_uris = [str(uri) for uri in mcp._resource_manager._resources.keys()]
        assert any("papers/latest" in uri for uri in resource_uris)

    def test_mcp_server_has_prompts(self):
        from src.mcp_server.server import mcp

        prompt_names = list(mcp._prompt_manager._prompts.keys())
        assert "physics_qa_template" in prompt_names
