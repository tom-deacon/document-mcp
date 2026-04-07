@echo off
cd /d C:\Development\GitHub\document-mcp

set MCP_TRANSPORT=http
set MCP_HOST=0.0.0.0
set MCP_PORT=8000

uv run python -m src.main