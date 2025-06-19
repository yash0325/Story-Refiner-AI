import streamlit as st
import os
from dotenv import load_dotenv
from jira import JIRA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import re

# --- ENV/SETUP ---
load_dotenv()
st.set_page_config(page_title="User Story Refiner AI", layout="wide")
st.title("📘 User Story Refiner AI")
st.caption("Powered by OpenAI GPT-4o")

# --- PROMPT TEMPLATE ---
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

def get_llm():
    return ChatOpenAI(model="gpt-4o", temperature=0)

def get_chain():
    return LLMChain(
        llm=get_llm(),
        prompt=PromptTemplate.from_template(REFINER_PROMPT)
    )

@st.cache_resource
def connect_to_jira():
    host = os.getenv("JIRA_HOST")
    email = os.getenv("JIRA_EMAIL")
    token = os.getenv("JIRA_API_TOKEN")
    if not all([host, email, token]):
        st.error("JIRA credentials missing in .env file.")
        return None
    return JIRA(server=host, basic_auth=(email, token))

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

jira = connect_to_jira()
project_key = os.getenv("JIRA_PROJECT_KEY")

if jira and project_key:
    st.success(f"Connected to JIRA: {project_key}")
    jql = f'project={project_key} ORDER BY created ASC'
    issues = jira.search_issues(jql, maxResults=20)

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
            st.markdown(f"**Summary:**\n{selected_issue.fields.summary}")
            st.markdown(f"**Description:**\n{selected_issue.fields.description or ''}")

        with col2:
            st.subheader("✨ Refined Output")

            # Stateless Form for Refinement
            with st.form("refine_form", clear_on_submit=True):
                submitted = st.form_submit_button("🔁 Refine Story")
                if submitted:
                    with st.spinner("Refining with AI..."):
                        chain = get_chain()
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
                        # --- Store most recent refined values in session state for update button
                        st.session_state["last_refined_summary"] = refined_summary
                        st.session_state["last_refined_criteria"] = refined_criteria
                        st.session_state["last_selected_issue_key"] = selected_issue.key

            # --- Show Update Jira button only if we have new output for this story
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
                        # Optionally clear state
                        # st.session_state["last_refined_summary"] = None
                        # st.session_state["last_refined_criteria"] = None
                    except Exception as e:
                        st.error(f"Failed to update Jira: {e}")

    else:
        st.warning("No issues found in the selected project.")
else:
    st.stop()
