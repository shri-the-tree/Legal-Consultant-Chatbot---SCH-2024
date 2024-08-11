import os
import time
import tempfile
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

def upload_to_gemini(uploaded_file, mime_type=None):
    """Uploads the given file to Gemini."""
    if uploaded_file is None:
        return None

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        # Write the contents of the uploaded file to the temporary file
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # Upload the temporary file to Gemini
        file = genai.upload_file(tmp_file_path, mime_type=mime_type)
        st.success(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file
    finally:
        # Clean up the temporary file
        os.unlink(tmp_file_path)

# Create the model
generation_config = {
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="You are a legal expert with access to the provided document. Answer questions about the case, summarize key points, and provide legal insights. Be concise and informative. Be as user friendly as possible.",
)

st.title("Legal Document Analysis Chatbot")
st.image("img.png")

st.markdown("""
<style>
body {
    font-family: 'Georgia', serif;
    color: #000000;
}
.stTitle {
    color: #A52A2A;
    font-size: 2.5em;
    text-align: center;
}
.stSidebar {
    background-color: #9A2A2A;
    color: #2c3e50;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h3><i class="fa fa-gavel"></i> Ask a Legal Question</h3>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #DAA06D, #C2B280);
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)



uploaded_file = st.file_uploader("Upload a case file (PDF)", type="pdf")

if uploaded_file is not None:
    file = upload_to_gemini(uploaded_file, mime_type="application/pdf")

    if file:
        st.success("File processed. You can now ask questions about the case.")

        try:
            chat_session = model.start_chat(
                history=[
                    {
                        "role": "user",
                        "parts": [file, "Analyze this document and provide a summary."],
                    },
                ]
            )



            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat messages from history on rerun
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # React to user input
            if prompt := st.chat_input("What would you like to know about the case?"):
                # Display user message in chat message container
                st.chat_message("user").markdown(prompt)
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})

                response = chat_session.send_message(prompt)

                # Check if the response contains text
                if response and hasattr(response, 'text') and response.text:
                    # Display assistant response in chat message container
                    with st.chat_message("assistant"):
                        st.markdown(response.text)
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                else:
                    st.error("No response received from the assistant.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

else:
    st.info("Please upload a case file to begin.")
