from github import Github, Auth
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import AgentOutput, ToolCall, ToolCallResult, FunctionAgent, AgentWorkflow
from llama_index.core.workflow import Context
from llama_index.core.prompts import RichPromptTemplate
from typing import Any
import os, dotenv, asyncio


# Initialize .env variables
dotenv.load_dotenv()


# Initialize LLM
llm = OpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
)

# Initialize github repo
auth = Auth.Token(os.getenv("GITHUB_TOKEN")) if os.getenv("GITHUB_TOKEN") else None
git = Github(auth=auth)
repo_url = f"https://github.com/{os.getenv("REPOSITORY")}.git"
repo_name = repo_url.split('/')[-1].replace('.git','')
username = repo_url.split('/')[-2]
full_repo_name = f"{username}/{repo_name}"
pr_number = os.getenv("PR_NUMBER")
if git is not None:
    repo = git.get_repo(full_repo_name)


# Tool definitions
async def get_file_contents(path: str):
    """Retrieve the contents of a file from the repository given the file path."""
    contents = repo.get_contents(path=path)
    return contents.decoded_content.decode("utf-8")


async def get_pr_details(pr_number: int) -> dict:
    """Retrieve details about a specific Pull Request given the PR number."""
    pr = repo.get_pull(number=pr_number)

    # Create a list of commits by SHA hash
    commit_hashes = []
    commits = pr.get_commits()
    for c in commits:
        commit_hashes.append(c.sha)

    pr_details = {"author": pr.user.login,
                  "title": pr.title,
                  "body": pr.body,
                  "diff_url": pr.diff_url,
                  "state": pr.state,
                  "head_sha": pr.head.sha,
                  "commits": commit_hashes,
                  }
    return pr_details


async def get_pr_commits(sha: str) -> list:
    """Use the commit SHA from the PR details tool and pass it to this tool to retrieve commit details."""
    commit = repo.get_commit(sha)
    changed_files: list[dict[str, Any]] = []
    for f in commit.files:
        changed_files.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "changes": f.changes,
            "patch": f.patch,
        })
    return changed_files


async def post_final_review(pr_number: int, final_review) -> str:
    pr = repo.get_pull(number=pr_number)
    pr.create_review(body=final_review, event="COMMENT")
    return "Posted the final review on GitHub."


async def save_summary(ctx: Context, summary: str) -> str:
    """Save the generated PR context summary to the shared state."""
    current_state = await ctx.store.get("state")
    current_state["context_summary"] = summary
    await ctx.store.set("state", current_state)
    return "Context summary saved to state."


async def save_draft_comment(ctx: Context, draft_comment: str) -> str:
    """Save a draft pull request comment to the shared state."""
    current_state = await ctx.store.get("state")
    current_state["draft_comment"] = draft_comment
    await ctx.store.set("state", current_state)
    return "Draft comment saved to state."


async def save_final_review(ctx: Context, final_review: str) -> str:
    """Save the final review to the shared state."""
    current_state = await ctx.store.get("state")
    current_state["final_review"] = final_review
    await ctx.store.set("state", current_state)
    return "Final review saved to state."


async def get_context_summary(ctx: Context) -> str:
    """Retrieve the context summary from the shared state."""
    state = await ctx.store.get("state")
    return state.get("context_summary") or "No summary available yet - request it from ContextAgent."


file_contents_tool = FunctionTool.from_defaults(get_file_contents)
get_pr_details_tool = FunctionTool.from_defaults(get_pr_details)
get_pr_commits_tool = FunctionTool.from_defaults(get_pr_commits)
save_summary_tool = FunctionTool.from_defaults(save_summary)
save_draft_comment_tool = FunctionTool.from_defaults(save_draft_comment)
get_context_summary_tool = FunctionTool.from_defaults(get_context_summary)
save_final_review_tool = FunctionTool.from_defaults(save_final_review)
post_final_review_tool = FunctionTool.from_defaults(post_final_review)


context_agent = FunctionAgent(
    llm=llm,
    name="ContextAgent",
    description="Gathers context information for other agents to utilize.",
    system_prompt="""
    You are the context gathering agent. When gathering context, you MUST gather:
        - The details: author, title, body, diff_url, state, and head_sha;
        - Changed files;
        - Any requested for files;
    Once you gather the requested info, write a concise summary of the PR with the above information. 
    Save this summary by calling the save_summary tool.
    After save_summary succeeds, you MUST hand control back to the Commentor Agent.
    """,
    tools=[file_contents_tool, get_pr_details_tool, get_pr_commits_tool, save_summary_tool],
    can_handoff_to=["CommentorAgent"]
)


commentor_agent = FunctionAgent(
    llm=llm,
    name="CommentorAgent",
    description="Uses the context gathered by the context agent to draft a pull review comment.",
    system_prompt="""
    You are the commentor agent that writes review comments for pull requests as a human reviewer would. 
    Ensure to do the following for a thorough review:
      - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent.
      - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing: 
        - What is good about the PR? 
        - Did the author follow ALL contribution rules? What is missing? 
        - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this. 
        - Are new endpoints documented? - use the diff to determine this. 
        - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement. 
      - Before drafting your review, call get_context_summary to retrieve the PR summary already gathered.
        if it is empty, you must hand off to ContextAgent first. 
      - You should directly address the author. So your comments should sound like: 
      "Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?"
      - You must hand off to the ReviewAndPostingAgent once you are done drafting a review.
    """,
    tools=[save_draft_comment_tool, get_context_summary_tool],
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"],
)


review_and_posting_agent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description="Reviews the comment generated by CommentorAgent and posts the final review to GitHub.",
    system_prompt="""
    You are the Review and Posting agent. You must use the CommentorAgent to create a review comment.
    Once a review is generated, you need to run a final check and post it to GitHub.
        - The review must: 
          - Be a ~200-300 word review in markdown format. 
          - Specify what is good about the PR. 
          - Did the author follow ALL contribution rules? What is missing? 
          - Are there notes on test availability for new functionality? If there are new models, are there migrations for them? 
          - Are there notes on whether new endpoints were documented? 
          - Are there suggestions on which lines could be improved upon? Are these lines quoted?
    If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. 
    When you are satisfied, post the review to GitHub.
    """,
    tools=[save_final_review_tool, post_final_review_tool],
    can_handoff_to=["CommentorAgent"],
)

workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_posting_agent],
    root_agent=review_and_posting_agent.name,
    initial_state={
        "context_summary": "",
        "draft_comment": "",
        "final_review": "",
    },
)

async def main():
    query = "Write a review for PR: " + pr_number
    prompt = RichPromptTemplate(query)

    handler = workflow_agent.run(user_msg=prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\n\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")


if __name__ == "__main__":
    asyncio.run(main())
    git.close()
