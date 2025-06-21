import streamlit as st
from jira import JIRA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import re
import os

st.set_page_config(page_title="User Story Refiner AI", layout="wide")
st.title("📘 User Story Refiner AI")

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

def clear_connection_state():
    for k in [
        "jira_host", "jira_email", "jira_api_token", "jira_project_key",
        "openai_api_key", "connected", "last_refined_summary",
        "last_refined_criteria", "last_selected_issue_key"
    ]:
        if k in st.session_state:
            del st.session_state[k]

# --- DISCONNECT BUTTON (ALWAYS AT TOP RIGHT IF CONNECTED) ---
if st.session_state.get("connected", False):
    colc, cold = st.columns([10, 1])
    with cold:
        if st.button("Disconnect"):
            clear_connection_state()
            st.rerun()

# ---- Step 1: Jira/OpenAI Connection Form ----
if not st.session_state.get("connected", False):
    st.subheader("Connect to Jira & OpenAI")
    with st.form("connection_form"):
        jira_host = st.text_input("Jira Host URL (e.g. https://yourdomain.atlassian.net)", value=st.session_state.get("jira_host", ""))
        jira_email = st.text_input("Jira Email", value=st.session_state.get("jira_email", ""))
        jira_api_token = st.text_input("Jira API Token", type="password", value=st.session_state.get("jira_api_token", ""))
        jira_project_key = st.text_input("Jira Project Key", value=st.session_state.get("jira_project_key", ""))
        openai_api_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.get("openai_api_key", ""))
        submitted = st.form_submit_button("Connect")

    if submitted:
        # Basic validation
        if not (jira_host and jira_email and jira_api_token and jira_project_key and openai_api_key):
            st.warning("Please fill in all fields to connect.")
        else:
            st.session_state["jira_host"] = jira_host.strip()
            st.session_state["jira_email"] = jira_email.strip()
            st.session_state["jira_api_token"] = jira_api_token.strip()
            st.session_state["jira_project_key"] = jira_project_key.strip()
            st.session_state["openai_api_key"] = openai_api_key.strip()
            # Try connection immediately
            try:
                jira = JIRA(server=jira_host, basic_auth=(jira_email, jira_api_token))
                st.session_state["connected"] = True
                st.success(f"Connected as {jira_email} to JIRA: {jira_project_key}")
            except Exception as e:
                st.session_state["connected"] = False
                st.error(f"Failed to connect to Jira: {e}")
else:
    # Connected UI banner
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
    openai_api_key = st.session_state["openai_api_key"]

    def get_llm(api_key):
        return ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key)

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

        filtered_issues = []
        issue_titles = []

        for i in issues:
            desc = i.fields.description or ""
            refined_flag = "_Refined by AI agent_" in desc
            if show_only_unrefined and refined_flag:
                continue
            label = f"{'✅ ' if refined_flag else ''}{i.key}: {i.fields.summary}"
            issue_titles.append(label)
            filtered_issues.append(i)

        if not issue_titles:
            st.warning("No unrefined stories found.")
            st.stop()

        selected = st.selectbox("Select a user story to refine:", issue_titles)
        selected_issue = filtered_issues[issue_titles.index(selected)]
        story_input = f"{selected_issue.fields.summary}\n\n{selected_issue.fields.description or ''}".strip()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📝 Original Story")
            st.markdown(f"**Summary:** {selected_issue.fields.summary}")
            st.markdown(f"**Description:** {selected_issue.fields.description or ''}")

        with col2:
            st.subheader("✨ Refined Output")
            with st.form("refine_form", clear_on_submit=True):
                submitted = st.form_submit_button("🔁 Refine Story")
                if submitted:
                    with st.spinner("Refining with AI..."):
                        chain = LLMChain(
                            llm=get_llm(openai_api_key),
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
    else:
        st.warning("No issues found in the selected project.")
