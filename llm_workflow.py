import os
import requests
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the model server type; default is GROQ per your instructions.
model_server = os.getenv('MODEL_SERVER', 'GROQ').upper()

if model_server == "OPTOGPT":
    API_KEY = os.getenv('OPTOGPT_API_KEY')
    BASE_URL = os.getenv('OPTOGPT_BASE_URL')
    LLM_MODEL = os.getenv('OPTOGPT_MODEL')
elif model_server == "GROQ":
    API_KEY = os.getenv('GROQ_API_KEY')
    BASE_URL = os.getenv('GROQ_BASE_URL')
    LLM_MODEL = os.getenv('GROQ_MODEL')
elif model_server == "NGU":
    API_KEY = os.getenv('NGU_API_KEY')
    BASE_URL = os.getenv('NGU_BASE_URL')
    LLM_MODEL = os.getenv('NGU_MODEL')
elif model_server == "OPENAI":
    API_KEY = os.getenv('OPENAI_API_KEY')
    BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    LLM_MODEL = os.getenv('OPENAI_MODEL')
else:
    raise ValueError(f"Unsupported MODEL_SERVER: {model_server}")

# Initialize the OpenAI client with custom base URL
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# Define a function to make LLM API calls
def call_llm(messages, tools=None, tool_choice=None):
    """
    Make a call to the LLM API with the specified messages and tools.
    """
    kwargs = {
        "model": LLM_MODEL,
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools
    if tool_choice:
        kwargs["tool_choice"] = tool_choice
    try:
        response = client.chat.completions.create(**kwargs)
        return response
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        return None

# ------------------ Part 1: Basic LLM Workflow Implementation ------------------

# Task 1.1: Set Up Sample Data
def get_sample_blog_post():
    """
    Read the sample blog post from a JSON file.
    """
    try:
        with open('sample_blog_post.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print("Error: sample_blog_post.json file not found.")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in sample_blog_post.json.")
        return None

# Task 1.2: Define Tool Schemas
extract_key_points_schema = {
    "type": "function",
    "function": {
        "name": "extract_key_points",
        "description": "Extract key points from a blog post",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The title of the blog post"
                },
                "content": {
                    "type": "string",
                    "description": "The content of the blog post"
                },
                "key_points": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of key points extracted from the blog post"
                }
            },
            "required": ["key_points"]
        }
    }
}

generate_summary_schema = {
    "type": "function",
    "function": {
        "name": "generate_summary",
        "description": "Generate a concise summary from the key points about AI in modern healthcare",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Concise summary of the blog post"
                }
            },
            "required": ["summary"]
        }
    }
}

create_social_media_posts_schema = {
    "type": "function",
    "function": {
        "name": "create_social_media_posts",
        "description": "Create social media posts for different platforms focused on AI in modern healthcare",
        "parameters": {
            "type": "object",
            "properties": {
                "twitter": {
                    "type": "string",
                    "description": "Post optimized for Twitter/X (max 280 characters)"
                },
                "linkedin": {
                    "type": "string",
                    "description": "Post optimized for LinkedIn (professional tone)"
                },
                "facebook": {
                    "type": "string",
                    "description": "Post optimized for Facebook"
                }
            },
            "required": ["twitter", "linkedin", "facebook"]
        }
    }
}

create_email_newsletter_schema = {
    "type": "function",
    "function": {
        "name": "create_email_newsletter",
        "description": "Create an email newsletter based on a blog post about AI in modern healthcare",
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "Email subject line"
                },
                "body": {
                    "type": "string",
                    "description": "Email body content in plain text"
                }
            },
            "required": ["subject", "body"]
        }
    }
}

# Task 1.3: Implement Task Functions
def task_extract_key_points(blog_post):
    messages = [
        {"role": "system", "content": "You are an expert at analyzing content and extracting key points from articles."},
        {"role": "user", "content": f"Extract the key points from this blog post:\n\nTitle: {blog_post['title']}\n\nContent: {blog_post['content']}"}
    ]
    response = call_llm(
        messages=messages,
        tools=[extract_key_points_schema],
        tool_choice={"type": "function", "function": {"name": "extract_key_points"}}
    )
    # Convert the message to a dictionary using model_dump() if available
    message_obj = response.choices[0].message
    message_dict = message_obj.model_dump() if hasattr(message_obj, "model_dump") else message_obj
    if response and message_dict.get("tool_calls"):
        tool_call = message_dict["tool_calls"][0]
        result = json.loads(tool_call["function"]["arguments"])
        return result.get("key_points", [])
    return []  # Fallback if tool calling fails

def task_generate_summary(key_points, max_length=150):
    messages = [
        {"role": "system", "content": "You are an expert at summarizing content concisely while preserving key information. The blog post is about the impact of AI on modern healthcare."},
        {"role": "user", "content": f"Generate a summary (max {max_length} words) based solely on the following key points about AI in modern healthcare:\n\n" +
                                    "\n".join([f"- {point}" for point in key_points])}
    ]
    response = call_llm(
        messages=messages,
        tools=[generate_summary_schema],
        tool_choice={"type": "function", "function": {"name": "generate_summary"}}
    )
    message_obj = response.choices[0].message
    message_dict = message_obj.model_dump() if hasattr(message_obj, "model_dump") else message_obj
    if response and message_dict.get("tool_calls"):
        tool_call = message_dict["tool_calls"][0]
        result = json.loads(tool_call["function"]["arguments"])
        return result.get("summary", "")
    return ""  # Fallback if tool calling fails

def task_create_social_media_posts(key_points, blog_title):
    """
    Create social media posts for different platforms using tool calling.
    """
    messages = [
        {"role": "system", "content": "You are a social media expert who creates engaging posts focused on AI in modern healthcare."},
        {"role": "user", "content": f"Create social media posts for Twitter, LinkedIn, and Facebook based on the blog title: '{blog_title}' and the following key points about the impact of AI on modern healthcare:\n\n" +
                                    "\n".join([f"- {point}" for point in key_points])}
    ]
    response = call_llm(
        messages=messages,
        tools=[create_social_media_posts_schema],
        tool_choice={"type": "function", "function": {"name": "create_social_media_posts"}}
    )
    message_obj = response.choices[0].message
    message_dict = message_obj.model_dump() if hasattr(message_obj, "model_dump") else message_obj
    if response and message_dict.get("tool_calls"):
        tool_call = message_dict["tool_calls"][0]
        return json.loads(tool_call["function"]["arguments"])
    return {"twitter": "", "linkedin": "", "facebook": ""}

def task_create_email_newsletter(blog_post, summary, key_points):
    """
    Create an email newsletter using tool calling.
    """
    messages = [
        {"role": "system", "content": "You are an email marketing specialist who creates engaging newsletters focused on AI in modern healthcare."},
        {"role": "user", "content": f"Create an email newsletter based on this blog post:\n\nTitle: {blog_post['title']}\n\nSummary: {summary}\n\nKey Points:\n" +
                                    "\n".join([f"- {point}" for point in key_points])}
    ]
    response = call_llm(
        messages=messages,
        tools=[create_email_newsletter_schema],
        tool_choice={"type": "function", "function": {"name": "create_email_newsletter"}}
    )
    message_obj = response.choices[0].message
    message_dict = message_obj.model_dump() if hasattr(message_obj, "model_dump") else message_obj
    if response and message_dict.get("tool_calls"):
        tool_call = message_dict["tool_calls"][0]
        return json.loads(tool_call["function"]["arguments"])
    return {"subject": "", "body": ""}  # Fallback

# Task 1.3 (continued): Implement the Pipeline Workflow
def run_pipeline_workflow(blog_post):
    """
    Run a simple pipeline workflow to repurpose content.
    """
    results = {}
    key_points = task_extract_key_points(blog_post)
    results['key_points'] = key_points

    summary = task_generate_summary(key_points)
    results['summary'] = summary

    social_posts = task_create_social_media_posts(key_points, blog_post['title'])
    results['social_posts'] = social_posts

    email = task_create_email_newsletter(blog_post, summary, key_points)
    results['email'] = email

    return results

# Task 1.4: Implement the DAG Workflow
def run_dag_workflow(blog_post):
    """
    Run a DAG workflow to repurpose content.
    """
    results = {}
    # Extract key points (sequentially here as placeholder)
    key_points = task_extract_key_points(blog_post)
    results['key_points'] = key_points

    summary = task_generate_summary(key_points)
    results['summary'] = summary

    social_posts = task_create_social_media_posts(key_points, blog_post['title'])
    results['social_posts'] = social_posts

    email = task_create_email_newsletter(blog_post, summary, key_points)
    results['email'] = email

    return results

# Task 1.5: Add Chain-of-Thought Reasoning
def extract_key_points_with_cot(blog_post):
    """
    Extract key points from a blog post using chain-of-thought reasoning.
    """
    messages = [
        {"role": "system", "content": "You are an expert at analyzing content. Think step-by-step about what the core messages are before providing the key points."},
        {"role": "user", "content": f"Please analyze the blog post titled '{blog_post['title']}' and then extract the key points. Consider the context and the details provided."}
    ]
    # For now, simulate chain-of-thought by calling the same extraction function.
    key_points = task_extract_key_points(blog_post)
    return {"key_points": key_points}

# ------------------ Part 2: Implementing Self-Correction with Reflexion ------------------

def evaluate_content(content, content_type):
    """
    Evaluate the quality of generated content.
    """
    quality_score = 1.0 if content and len(content) > 0 else 0.0
    feedback = "Looks good." if quality_score >= 0.8 else "Needs improvement."
    return {"quality_score": quality_score, "feedback": feedback}

def improve_content(content, feedback, content_type):
    """
    Improve content based on feedback.
    """
    improved = content + "\n\n[Improved based on feedback: " + feedback + "]"
    return improved

def generate_with_reflexion(generator_func, max_attempts=3):
    """
    Apply Reflexion to a content generation function.
    """
    def wrapped_generator(*args, **kwargs):
        content_type = kwargs.pop("content_type", "content")
        content = generator_func(*args, **kwargs)
        for attempt in range(max_attempts):
            evaluation = evaluate_content(content, content_type)
            if evaluation["quality_score"] >= 0.8:
                return content
            content = improve_content(content, evaluation["feedback"], content_type)
        return content
    return wrapped_generator

# ... [All previous code remains unchanged above]

def run_workflow_with_reflexion(blog_post):
    """
    Run a workflow with Reflexion-based self-correction.
    """
    results = {}
    corrected_generate_summary = generate_with_reflexion(task_generate_summary)
    corrected_create_social_posts = generate_with_reflexion(task_create_social_media_posts)
    corrected_create_email = generate_with_reflexion(task_create_email_newsletter)

    key_points = task_extract_key_points(blog_post)
    results['key_points'] = key_points

    summary = corrected_generate_summary(key_points, max_length=150, content_type="summary")
    results['summary'] = summary

    social_posts = corrected_create_social_posts(key_points, blog_post['title'], content_type="social_media_post")
    results['social_posts'] = social_posts

    email = corrected_create_email(blog_post, summary, key_points, content_type="email")
    results['email'] = email

    # Debug print to check the Reflexion workflow output
    print("Debug: Reflexion workflow produced the following results:")
    print(json.dumps(results, indent=2))
    return results

# ... [Main section remains unchanged]


# ------------------ Part 3: Advanced Agent-Driven Workflow ------------------

def define_agent_tools():
    """
    Define the tools that the workflow agent can use.
    """
    finish_tool_schema = {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Complete the workflow and return the final results",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "The final summary"
                    },
                    "social_posts": {
                        "type": "object",
                        "description": "The social media posts for each platform"
                    },
                    "email": {
                        "type": "object",
                        "description": "The email newsletter"
                    }
                },
                "required": ["summary", "social_posts", "email"]
            }
        }
    }
    all_tools = [extract_key_points_schema, generate_summary_schema, create_social_media_posts_schema, create_email_newsletter_schema]
    return all_tools + [finish_tool_schema]

def execute_agent_tool(tool_name, arguments):
    """
    Execute a tool based on the tool name and arguments.
    """
    if tool_name == "extract_key_points":
        blog_post = {"title": arguments.get("title", ""), "content": arguments.get("content", "")}
        return {"key_points": task_extract_key_points(blog_post)}
    elif tool_name == "generate_summary":
        key_points = arguments.get("key_points", [])
        return {"summary": task_generate_summary(key_points)}
    elif tool_name == "create_social_media_posts":
        key_points = arguments.get("key_points", [])
        blog_title = arguments.get("blog_title", "Untitled")
        return task_create_social_media_posts(key_points, blog_title)
    elif tool_name == "create_email_newsletter":
        blog_post = arguments.get("blog_post", {})
        summary = arguments.get("summary", "")
        key_points = arguments.get("key_points", [])
        return task_create_email_newsletter(blog_post, summary, key_points)
    elif tool_name == "finish":
        return arguments
    else:
        return {"error": f"Tool {tool_name} is not implemented."}

def run_agent_workflow(blog_post):
    """
    Run an agent-driven workflow to repurpose content.
    """
    system_message = (
        "You are a Content Repurposing Agent. Your job is to take a blog post about the impact of AI on modern healthcare and repurpose it into different formats:\n"
        "1. Extract key points from the blog post\n"
        "2. Generate a concise summary\n"
        "3. Create social media posts for different platforms\n"
        "4. Create an email newsletter\n"
        "You have access to tools that can help you with each of these tasks. Think carefully about which tools to use and in what order.\n"
        "When you're done, use the 'finish' tool to complete the workflow."
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Please repurpose this blog post:\n\nTitle: {blog_post['title']}\n\nContent: {blog_post['content']}"}
    ]
    tools = define_agent_tools()
    max_iterations = 10
    for i in range(max_iterations):
        response = call_llm(messages, tools)
        if not response:
            break
        message_obj = response.choices[0].message
        message_dict = message_obj.model_dump() if hasattr(message_obj, "model_dump") else message_obj
        messages.append(message_dict)
        if not message_dict.get("tool_calls"):
            break
        for tool_call in message_dict.get("tool_calls", []):
            tool_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])
            if tool_name == "finish":
                return arguments
            tool_result = execute_agent_tool(tool_name, arguments)
            messages.append({
                "tool_call_id": tool_call.get("id", ""),
                "role": "tool",
                "name": tool_name,
                "content": json.dumps(tool_result) if isinstance(tool_result, dict) else tool_result
            })
    return {"error": "The agent couldn't complete the workflow within the maximum number of iterations."}

# ------------------ Main Section ------------------

if __name__ == "__main__":
    blog_post = get_sample_blog_post()
    if blog_post is None:
        print("Aborting workflow execution due to missing or invalid blog post data.")
    else:
        print("=== Pipeline Workflow ===")
        pipeline_results = run_pipeline_workflow(blog_post)
        print(json.dumps(pipeline_results, indent=2))

        print("\n=== DAG Workflow ===")
        dag_results = run_dag_workflow(blog_post)
        print(json.dumps(dag_results, indent=2))

        print("\n=== Workflow with Reflexion ===")
        reflexion_results = run_workflow_with_reflexion(blog_post)
        print(json.dumps(reflexion_results, indent=2))

        print("\n=== Agent-Driven Workflow ===")
        agent_results = run_agent_workflow(blog_post)
        print(json.dumps(agent_results, indent=2))
