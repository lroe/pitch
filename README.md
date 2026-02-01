Pitchine: AI Pitch Simulator
Pitchine is an AI-powered simulation environment designed for startup founders to hone their pitching skills. By simulating various "Investor Personas" with distinct biases and questioning styles, Pitchine provides a high-stakes, low-risk sandbox for entrepreneurs to refine their narrative, handle objections, and master their data.

## Core Philosophy & Critique
While most pitch trainers focus on "soft skills" like tone and pace, Pitchine focuses on substantive rigor. The AI is programmed to identify logical fallacies, market inconsistencies, and weak unit economics.

Challenge: Most founders fail not because of their delivery, but because their underlying assumptions don't hold up under scrutiny. Pitchine is built to stress-test those assumptions.

## Tech Stack
Backend: Python (FastAPI/Flask) - Handles the LLM orchestration and logic.

Database & Auth: Supabase - Manages user profiles, pitch transcripts, and historical performance data.

Frontend: JavaScript (React/Next.js) - Provides a real-time, interactive interface for the simulation.

AI Engine: Integration with LLMs (OpenAI/Anthropic) via LangChain or direct API.

## Features
Persona Selection: Choose from "The Skeptical VC," "The Technical Specialist," or "The Aggressive Shark."

Real-time Feedback: Instant analysis of your responses using NLP to gauge clarity and confidence.

Session Persistence: All sessions are stored in Supabase, allowing you to track your improvement over time.

The "Grill" Mode: An automated rapid-fire Q&A session specifically designed to find the holes in your business model.

## Getting Started
### Prerequisites
Python 3.9+

Node.js & npm

Supabase Account
