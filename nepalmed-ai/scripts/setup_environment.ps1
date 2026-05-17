# Setup script for NepalMed AI dependencies
# Run this in your terminal to install the necessary libraries

Write-Host "Installing NepalMed AI Dependencies..." -ForegroundColor Cyan

pip install ollama chromadb langchain-ollama langchain-community pymupdf sentence-transformers pydicom

Write-Host "✅ Installation complete." -ForegroundColor Green
Write-Host "Note: Ensure you have Ollama installed on your system (https://ollama.com/)" -ForegroundColor Yellow
