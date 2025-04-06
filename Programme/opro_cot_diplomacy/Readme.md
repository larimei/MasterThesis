# Diplomacy Game Communication Analysis Tool

## 🎲 Project Overview

This tool provides an advanced analysis of Diplomacy game communications, utilizing AI to optimize and understand player interactions across different game phases.

## 🚀 Features

- Analyze game communication strategies
- Optimize diplomatic messages
- Generate comprehensive game summaries
- Identify key communication patterns
- Evaluate trust and strategic moves

## 🛠️ Installation Guide

### 1. Clone the Repository

```bash
git clone https://github.com/larimei/MasterThesis
cd opro_cot_diplomacy
```

### 2. Create Virtual Environment

#### Windows

```bash
python -m venv diplomacy_env
diplomacy_env\Scripts\activate
```

### 3. Install Ollama

#### Windows/macOS

1. Download from [Ollama Official Website](https://ollama.com/download)
2. Follow installation instructions

### 4. Pull Language Model

```bash
ollama pull llama3.1:8b
ollama pull deepseek-r1:8b
```

## 🎮 Usage

### Run the Analysis

- Ensure Ollama is running: `ollama serve`

```bash
python main.py
```

### Interactive Prompts

- You'll be asked to provide:
  1. Input JSON file path
  2. Output directory for results

## 📂 Project Structure

- `main.py`: Main script
- `prompt_utils_gemini.py`: LLM interaction utilities
- `outputs/`: Directory for analysis results
- `data/`: Directory for diplomacy input json files
