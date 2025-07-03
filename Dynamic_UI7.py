import streamlit as st
from jira import JIRA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

st.set_page_config(page_title="User Story Refiner AI", layout="wide")
st.title("📘 User Story Refiner AI")

# ---- PROMPTS ----
REFINER_PROMPT = """
You are a User Story Refiner Agent. Given a user story or backlog item (which may be unclear, incomplete, or poorly written), your tasks are:
1. Rewrite the story for clarity and completeness using the INVEST criteria.
2. Add actionable acceptance criteria in bullet points.
3. Suggest improvements if information is missing.

Input User Story/Backlog Item:
{user_story}

Output (in this format):
---
**Refined User Story:**  
<improved version>

**Acceptance Criteria:**  
- <criterion 1>
- <criterion 2>
---
"""

TASK_BREAKDOWN_PROMPT = """
You are a software analyst. Given the following user story and its acceptance criteria, break it down into a clear, actionable list of implementation tasks for the development team.

User Story:
{user_story}

Acceptance Criteria:
{acceptance_criteria}

Output (as a bullet list of tasks):
-
"""

# ---- JIRA SUB-TASK HELPERS ----
def get_subtask_issue_type(jira, project_key):
    """Get the sub-task issue type name for the project."""
    project = jira.project(project_key)
    for issue_type in project.issueTypes:
        if issue_type.subtask:
            return issue_type.name
    raise Exception("Sub-task issue type not found in this project!")

def create_jira_subtask(jira, parent_issue_key, summary, project_key, subtask_issue_type):
    """Create a sub-task under the specified parent."""
    parent_issue = jira.issue(parent_issue_key)
    if parent_issue.fields.issuetype.subtask:
        raise Exception("Cannot create a sub-task under a sub-task!")
    issue_dict = {
        'project': {'key': project_key},
        'parent': {'key': parent_issue_key},
        'summary': summary[:255],
        'description': '',
        'issuetype': {'name': subtask_issue_type},
    }
    return jira.create_issue(fields=issue_dict)

def delete_existing_subtasks(jira, parent_issue_key):
    """Delete all sub-tasks under the specified parent issue."""
    parent_issue = jira.issue(parent_issue_key)
    subtask_keys = [subtask.key for subtask in parent_issue.fields.subtasks]
    for key in subtask_keys:
        try:
            jira.delete_issue(key)
        except Exception as e:
            st.error(f"Failed to delete sub-task {key}: {e}")

def clear_connection_state():
    for k in [
        "jira_host", "jira_email", "jira_api_token", "jira_project_key",
        "connected", "last_refined_summary",
        "last_refined_criteria", "last_selected_issue_key", "last_task_breakdown", "last_task_breakdown_lines"
    ]:
        if k in st.session_state:
            del st.session_state[k]

def parse_task_lines(task_lines):
    """Filter out headings (like 'User Signup and Password Management:') and only return actual sub-tasks."""
    parsed = []
    for line in task_lines:
        # Remove checkbox formatting, spaces, asterisks, and dashes
        clean = line.strip().lstrip("-•").strip()
        # Ignore headings (e.g. those ending with ':' or are all bold/uppercase or empty)
        if not clean:
            continue
        if clean.endswith(":"):
            continue
        if clean.startswith("**") and clean.endswith("**"):
            continue
        if len(clean.split()) <= 4 and clean == clean.upper():
            continue
        parsed.append(clean)
    return parsed

# --- DISCONNECT BUTTON (TOP RIGHT IF CONNECTED) ---
if st.session_state.get("connected", False):
    colc, cold = st.columns([10, 1])
    with cold:
        if st.button("Disconnect"):
            clear_connection_state()
            st.rerun()

# ---- Step 1: Jira Connection Form ----
if not st.session_state.get("connected", False):
    st.subheader("Connect to Jira")
    with st.form("connection_form"):
        jira_host = st.text_input("Jira Host URL (e.g. https://yourdomain.atlassian.net)", value=st.session_state.get("jira_host", ""))
        jira_email = st.text_input("Jira Email", value=st.session_state.get("jira_email", ""))
        jira_api_token = st.text_input("Jira API Token", type="password", value=st.session_state.get("jira_api_token", ""))
        jira_project_key = st.text_input("Jira Project Key", value=st.session_state.get("jira_project_key", ""))
        submitted = st.form_submit_button("Connect")

    if submitted:
        if not (jira_host and jira_email and jira_api_token and jira_project_key):
            st.warning("Please fill in all fields to connect.")
        else:
            st.session_state["jira_host"] = jira_host.strip()
            st.session_state["jira_email"] = jira_email.strip()
            st.session_state["jira_api_token"] = jira_api_token.strip()
            st.session_state["jira_project_key"] = jira_project_key.strip()
            try:
                jira = JIRA(server=jira_host, basic_auth=(jira_email, jira_api_token))
                st.session_state["connected"] = True
                st.success(f"Connected as {jira_email} to JIRA: {jira_project_key}")
            except Exception as e:
                st.session_state["connected"] = False
                st.error(f"Failed to connect to Jira: {e}")
else:
    st.success(
        f"Connected as {st.session_state['jira_email']} to JIRA: {st.session_state['jira_project_key']}",
        icon="🔗"
    )

# Only continue if connected
if st.session_state.get("connected", False):
    jira_host = st.session_state["jira_host"]
    jira_email = st.session_state["jira_email"]
    jira_api_token = st.session_state["jira_api_token"]
    jira_project_key = st.session_state["jira_project_key"]

    def get_llm():
        return ChatOpenAI(model="gpt-4o", temperature=0, api_key=st.secrets["OPENAI_API_KEY"])

    def parse_refined_output(output):
        lines = output.splitlines()
        refined_summary_lines = []
        refined_criteria_lines = []
        mode = None
        for line in lines:
            if "**Refined User Story:**" in line:
                mode = "summary"
                continue
            if "**Acceptance Criteria:**" in line:
                mode = "criteria"
                continue
            if line.strip() == "---":
                mode = None
                continue
            if mode == "summary":
                refined_summary_lines.append(line.strip())
            elif mode == "criteria":
                refined_criteria_lines.append(line.strip())
        return (
            " ".join(refined_summary_lines).strip(),
            "\n".join(refined_criteria_lines).strip()
        )

    try:
        jira = JIRA(server=jira_host, basic_auth=(jira_email, jira_api_token))
        jql = f'project={jira_project_key} ORDER BY created ASC'
        issues = jira.search_issues(jql, maxResults=20)
    except Exception as e:
        st.error(f"Failed to load issues: {e}")
        issues = []

    if issues:
        show_only_unrefined = st.checkbox("Show only unrefined stories", value=False)

        # Only allow valid parent issue types
        valid_parent_types = ["Story", "Task", "Bug"]
        filtered_issues = []
        issue_titles = []

        for i in issues:
            desc = i.fields.description or ""
            refined_flag = "_Refined by AI agent_" in desc
            # Only add if issue type is in allowed list
            if i.fields.issuetype.name not in valid_parent_types:
                continue
            if show_only_unrefined and refined_flag:
                continue
            label = f"{'✅ ' if refined_flag else ''}{i.key}: {i.fields.summary} ({i.fields.issuetype.name})"
            issue_titles.append(label)
            filtered_issues.append(i)

        if not issue_titles:
            st.warning("No unrefined stories found or no valid parent issues available. Only Story, Task, or Bug can have sub-tasks.")
            st.stop()

        selected = st.selectbox("Select a user story to refine:", issue_titles)
        selected_issue = filtered_issues[issue_titles.index(selected)]
        story_input = f"{selected_issue.fields.summary}\n\n{selected_issue.fields.description or ''}".strip()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📝 Original Story")
            st.markdown(f"**Summary:** {selected_issue.fields.summary}")
            st.markdown(f"**Description:** {selected_issue.fields.description or ''}")
            st.markdown(f"**Issue Type:** {selected_issue.fields.issuetype.name}")

        with col2:
            st.subheader("✨ Refined Output")
            with st.form("refine_form", clear_on_submit=True):
                submitted = st.form_submit_button("🔁 Refine Story")
                if submitted:
                    with st.spinner("Refining with AI..."):
                        chain = LLMChain(
                            llm=get_llm(),
                            prompt=PromptTemplate.from_template(REFINER_PROMPT)
                        )
                        try:
                            refined = chain.run({"user_story": story_input})
                        except Exception as e:
                            st.error(f"OpenAI Error: {e}")
                            refined = ""
                        refined_summary, refined_criteria = parse_refined_output(refined)
                        st.markdown(f"**Refined Summary:** {refined_summary}")
                        st.markdown("**Acceptance Criteria:**")
                        if "**Suggestions for Improvement:**" in refined_criteria:
                            criteria_part, suggestions_part = refined_criteria.split("**Suggestions for Improvement:**", 1)
                            st.markdown(f"- " + "\n- ".join([line for line in criteria_part.strip().splitlines() if line]))
                            st.markdown("**Suggestions for Improvement:**")
                            st.markdown(f"- " + "\n- ".join([line for line in suggestions_part.strip().splitlines() if line]))
                        else:
                            st.markdown(f"- " + "\n- ".join([line for line in refined_criteria.strip().splitlines() if line]))
                        # Store for update
                        st.session_state["last_refined_summary"] = refined_summary
                        st.session_state["last_refined_criteria"] = refined_criteria
                        st.session_state["last_selected_issue_key"] = selected_issue.key

            # Show Update Jira if a refined output is present for this story
            if (
                st.session_state.get("last_refined_summary")
                and st.session_state.get("last_selected_issue_key") == selected_issue.key
            ):
                if st.button("📌 Update Jira", key="update_jira_btn"):
                    refined_description = (
                        f"**Refined User Story:**  {st.session_state['last_refined_summary']}\n\n"
                        f"**Acceptance Criteria:**  \n"
                        f"- " + "\n- ".join(st.session_state['last_refined_criteria'].splitlines()) +
                        "\n\n_Refined by AI agent_"
                    )
                    try:
                        jira.issue(selected_issue.key).update(
                            summary=st.session_state['last_refined_summary'][:255],
                            description=refined_description
                        )
                        st.success(f"Issue {selected_issue.key} updated in Jira!")
                    except Exception as e:
                        st.error(f"Failed to update Jira: {e}")

            # --------- BREAK DOWN INTO TASKS FEATURE ---------
            if (
                st.session_state.get("last_refined_summary")
                and st.session_state.get("last_selected_issue_key") == selected_issue.key
            ):
                if st.button("🛠️ Break Down Into Tasks"):
                    with st.spinner("Breaking down into tasks..."):
                        chain = LLMChain(
                            llm=get_llm(),
                            prompt=PromptTemplate.from_template(TASK_BREAKDOWN_PROMPT)
                        )
                        tasks_output = chain.run({
                            "user_story": st.session_state["last_refined_summary"],
                            "acceptance_criteria": st.session_state["last_refined_criteria"]
                        })
                        st.markdown("**Implementation Tasks:**")
                        # Split by lines and remove any lines that are section headings or not sub-tasks
                        task_lines = [line.lstrip('-•').strip() for line in tasks_output.strip().splitlines() if line.strip()]
                        # Only keep actual sub-tasks, not headings
                        filtered_task_lines = parse_task_lines(task_lines)
                        for task in filtered_task_lines:
                            st.checkbox(task, key=task)
                        st.session_state["last_task_breakdown"] = "\n".join([f"- [ ] {task}" for task in filtered_task_lines])
                        st.session_state["last_task_breakdown_lines"] = filtered_task_lines  # <-- Store only the valid sub-tasks

                # ------ BUTTON TO CREATE SUB-TASKS ------
                if st.session_state.get("last_task_breakdown_lines"):
                    confirm_delete = st.checkbox(
                        "I understand this will delete ALL existing sub-tasks and replace them with the latest AI-generated ones.",
                        key="confirm_delete_subtasks"
                    )
                    if st.button("📎 Create Jira Sub-tasks (replace existing)", key="create_jira_subtasks_btn"):
                        if confirm_delete:
                            try:
                                subtask_issue_type = get_subtask_issue_type(jira, jira_project_key)
                                parent_issue_key = selected_issue.key
                                # Delete existing sub-tasks first
                                delete_existing_subtasks(jira, parent_issue_key)
                                created_keys = []
                                for task_summary in st.session_state["last_task_breakdown_lines"]:
                                    new_issue = create_jira_subtask(
                                        jira,
                                        parent_issue_key=parent_issue_key,
                                        summary=task_summary,
                                        project_key=jira_project_key,
                                        subtask_issue_type=subtask_issue_type,
                                    )
                                    created_keys.append(new_issue.key)
                                st.success(f"Created sub-tasks: {', '.join(created_keys)} (replacing any previous ones)")
                            except Exception as e:
                                st.error(f"Failed to create sub-tasks: {e}")
                        else:
                            st.warning("Please check the confirmation box before replacing sub-tasks!")

                # ------ Optional: Also keep "Update Jira with Tasks" if you want old behavior ------
                if st.session_state.get("last_task_breakdown"):
                    if st.button("📋 Update Jira with Tasks", key="update_jira_tasks_btn"):
                        refined_description = (
                            f"**Refined User Story:**  {st.session_state['last_refined_summary']}\n\n"
                            f"**Acceptance Criteria:**  \n"
                            f"- " + "\n- ".join(st.session_state['last_refined_criteria'].splitlines()) +
                            "\n\n**Implementation Tasks:**\n" +
                            st.session_state["last_task_breakdown"] +
                            "\n\n_Refined and broken down by AI agent_"
                        )
                        try:
                            jira.issue(selected_issue.key).update(
                                summary=st.session_state['last_refined_summary'][:255],
                                description=refined_description
                            )
                            st.success(f"Issue {selected_issue.key} updated in Jira with tasks!")
                        except Exception as e:
                            st.error(f"Failed to update Jira: {e}")

    else:
        st.warning("No issues found in the selected project or no eligible parent issues (Story, Task, or Bug).")
