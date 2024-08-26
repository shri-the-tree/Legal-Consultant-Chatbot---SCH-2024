import os
import tempfile
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import PyPDF2
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)


def extract_text_from_pdf(file):
    """Extracts text from a PDF file."""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def preprocess_text(text):
    """Preprocesses the extracted text."""
    # Implement text cleaning and normalization here
    return text


def upload_to_gemini(uploaded_file, mime_type=None):
    """Uploads the given file to Gemini."""
    if uploaded_file is None:
        return None

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        file = genai.upload_file(tmp_file_path, mime_type=mime_type)
        st.success(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file
    finally:
        os.unlink(tmp_file_path)


def generate_stakeholder_graph(text):
    """Generates a stakeholder relationship graph."""
    # This is a placeholder. In a real implementation, you'd use NLP to extract entities and relationships.
    G = nx.Graph()
    G.add_edge("Plaintiff", "Defendant")
    G.add_edge("Plaintiff", "Judge")
    G.add_edge("Defendant", "Judge")

    plt.figure(figsize=(10, 7))
    nx.draw(G, with_labels=True, node_color='lightblue', node_size=3000, font_size=16, font_weight='bold')
    plt.title("Stakeholder Relationships")
    return plt


def generate_timeline(text):
    """Generates a timeline of events."""
    # This is a placeholder. In a real implementation, you'd use NLP to extract dates and events.
    events = [
        ("2022-01-01", "Case Filed"),
        ("2022-03-15", "First Hearing"),
        ("2022-06-30", "Evidence Submission Deadline"),
        ("2022-09-01", "Trial Begins")
    ]

    fig = go.Figure(data=[go.Scatter(
        x=[event[0] for event in events],
        y=[event[1] for event in events],
        mode='markers+text',
        text=[event[1] for event in events],
        textposition='top center'
    )])
    fig.update_layout(title='Case Timeline', xaxis_title='Date', yaxis_title='Event')
    return fig


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
.stApp {
    background: linear-gradient(to right, #DAA06D, #C2B280);
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h3><i class="fa fa-gavel"></i> Ask a Legal Question</h3>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a case file (PDF)", type="pdf")

if uploaded_file is not None:
    file = upload_to_gemini(uploaded_file, mime_type="application/pdf")

    if file:
        st.success("File processed. You can now ask questions about the case.")

        try:
            # Extract and preprocess text
            raw_text = extract_text_from_pdf(uploaded_file)
            processed_text = preprocess_text(raw_text)

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
                st.chat_message("user").markdown(prompt)
                st.session_state.messages.append({"role": "user", "content": prompt})

                response = chat_session.send_message(prompt)

                if response and hasattr(response, 'text') and response.text:
                    with st.chat_message("assistant"):
                        st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                else:
                    st.error("No response received from the assistant.")

            # Visualization section
            st.sidebar.header("Visualizations")
            if st.sidebar.button("Generate Stakeholder Graph"):
                graph = generate_stakeholder_graph(processed_text)
                st.sidebar.pyplot(graph)

            if st.sidebar.button("Generate Timeline"):
                timeline = generate_timeline(processed_text)
                st.sidebar.plotly_chart(timeline)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

else:
    st.info("Please upload a case file to begin.")
