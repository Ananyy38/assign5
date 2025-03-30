# LLM Workflow Project

This repository contains a Python implementation of various LLM workflows that repurpose a blog post into different formats (key points, summary, social media posts, and an email newsletter).

## Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/llm-workflow-project.git
   ```
2. **Navigate to the Project Directory:**
   ```bash
   cd llm-workflow-project
   ```
3. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate
   ```
4. **Install Required Packages:**
   ```bash
   pip install -r requirements.txt
   ```
5. **Configure Environment Variables:**
   - Create a `.env` file in the project root.
   - Add your API keys, model server settings, etc.

## Implementation Overview

This project implements four workflow approaches:

- **Pipeline Workflow:** Sequentially processes tasks.
- **DAG Workflow:** Processes tasks in a directed acyclic graph structure.
- **Workflow with Reflexion:** Uses self-correction to improve generated content.
- **Agent-Driven Workflow:** An agent dynamically selects tools to complete the workflow.

Each workflow repurposes a blog post (about the impact of AI on modern healthcare) into:
- Extracted key points
- A concise summary
- Social media posts for Twitter, LinkedIn, and Facebook
- An email newsletter

## Example Outputs

Each workflow outputs JSON data. For example, a typical output might look like:

```json
{
  "key_points": ["AI is transforming healthcare through machine learning, NLP, and robotics."],
  "summary": "AI is revolutionizing modern healthcare by enhancing diagnostics and patient care...",
  "social_posts": {
    "twitter": "AI is transforming healthcare! #AI #HealthTech",
    "linkedin": "Discover the impact of AI on modern healthcare...",
    "facebook": "Explore how AI is revolutionizing healthcare..."
  },
  "email": {
    "subject": "Revolutionizing Healthcare with AI",
    "body": "Dear Subscriber,\n\nAI is changing modern healthcare by..."
  }
}
```

## Analysis & Challenges

- **Effectiveness:**  
  The Pipeline and DAG workflows provide reliable outputs, while the Reflexion and Agent-Driven workflows allow for dynamic improvements and flexible task execution.
  
- **Challenges:**  
  - Converting Pydantic models to dictionaries using `model_dump()` to avoid attribute errors.
  - Ensuring context consistency across workflows.
  - Fine-tuning prompt instructions to maintain topic relevance.

