import streamlit as st
import requests

# 1. CRITICAL: set_page_config must be the very first Streamlit command
st.set_page_config(page_title="AlphaLens Research Memo", page_icon="📊", layout="wide")

from Chatbot.run_chatbot import run_chatbot
from components import (
    render_header,
    render_summary,
    render_thesis,
    render_section_tabs,
    render_valuation_chart,
    render_key_risks,
    render_conflicting_signals,
    render_data_gaps,
    render_confidence_gauge,
)

API_URL = "https://alphalens-multi-ai-agent-stock-researcher.onrender.com"

# 2. Initialize Session State Variables
if "memo" not in st.session_state:
    st.session_state.memo = None
if "ui_chat_history" not in st.session_state:
    st.session_state.ui_chat_history = []
if "chosen_ticker" not in st.session_state:
    st.session_state.chosen_ticker = None
if "company_name" not in st.session_state:
    st.session_state.company_name = None

def main():
    st.title("AlphaLens Research")

    # Search Bar
    with st.form("search"):
        col1, col2 = st.columns([5, 1])
        with col1:
            query = st.text_input(
                "Research Query",
                placeholder="Enter a company name...",
                label_visibility="collapsed"
            )
        with col2:
            submit = st.form_submit_button("Analyze", use_container_width=True)

    # 3. Handle the API Call and save to session_state
    if submit and query:
        with st.spinner("Analyzing..."):
            try:
                response= requests.post(
                    f"{API_URL}/search_company", 
                    json={"userquery": query}  
                    )
                response.raise_for_status() # Catch HTTP errors
                st.session_state.chosen_ticker = None
                st.session_state.memo = None
                st.session_state.company_name = response.json()
            except Exception as e:
                st.error(f"Failed to fetch data: {e}")
                return

    if st.session_state.company_name :
        if st.session_state.company_name['status'] == "resolved":
            chosen_ticker = st.session_state.company_name['ticker']
            st.session_state.chosen_ticker= chosen_ticker
            st.rerun()

        elif st.session_state.company_name['status'] == "needs_confirmation":
            st.warning("Multiple matches found. Please select the correct company:")
        
            options = {
                f"{c['name']} ({c['ticker']}) - [{c['exchange']}]": c['ticker'] 
                for c in st.session_state.company_name['candidates']
            }

            selected_display = st.radio(
            "Select Target Company:", 
            options=list(options.keys())
            )

            if st.button("Confirm Selection", type="primary"):
                st.session_state.chosen_ticker = options[selected_display]
                st.rerun()
    
    if st.session_state.chosen_ticker:
        with st.spinner(f"Running full analysis for {st.session_state.chosen_ticker}..."):
            try:
                response_memo = requests.post(
                    f"{API_URL}/research", 
                    json={"userquery": st.session_state.chosen_ticker}
                )
                response_memo.raise_for_status()
                st.session_state.memo = response_memo.json()
            except Exception as e:
                st.error(f"Failed to generate memo: {e}")


    # 4. Only render the dashboard and chat if a memo exists
    if st.session_state.memo:
        memo_data = st.session_state.memo
        
        # Render Dashboard
        render_header(memo_data)
        st.divider()

        left, right = st.columns([3, 2])
        with left:
            render_summary(memo_data)
            render_thesis(memo_data)
            st.divider()
            render_section_tabs(memo_data)
            st.divider()
            render_valuation_chart(memo_data)

        with right:
            render_confidence_gauge(memo_data)
            render_key_risks(memo_data)
            render_conflicting_signals(memo_data)
            render_data_gaps(memo_data)

        # Render Chatbot Sidebar
        with st.sidebar:
            st.header("AI Bot")
            
            for msg in st.session_state.ui_chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Chat Input
        if user_input := st.chat_input("Type your message here..."):
            
            # Add user message to state and display
            st.session_state.ui_chat_history.append({"role": "user", "content": user_input})
            with st.sidebar:
                with st.chat_message("user"):
                    st.markdown(user_input)
                    
            # Generate and display bot response
            with st.spinner("Thinking..."):
                # Pass data safely from session state
                bot_response = run_chatbot(
                    str(memo_data), 
                    memo_data.get('ticker', 'UNKNOWN'), 
                    user_input
                )
            
            st.session_state.ui_chat_history.append({"role": "assistant", "content": bot_response})
            with st.sidebar:
                with st.chat_message("assistant"):
                    st.markdown(bot_response)
    else:
        st.info("Please enter a research query to generate the memo.")

if __name__ == "__main__":
    main()
