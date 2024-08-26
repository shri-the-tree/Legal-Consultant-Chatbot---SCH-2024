import os
import tempfile
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import PyPDF2
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import spacy
from collections import Counter
import pandas as pd

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


def extract_text_from_pdf(file):
    """Extracts text from a PDF file."""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def preprocess_text(text):
    """Preprocesses the extracted text."""
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)


def extract_entities(text):
    """Extracts named entities from the text."""
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities


def extract_dates(text):
    """Extracts dates from the text."""
    doc = nlp(text)
    dates = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ == "DATE"]
    return dates


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


def generate_stakeholder_graph(entities):
    """Generates a stakeholder relationship graph."""
    G = nx.Graph()
    person_entities = [ent for ent, label in entities if label == "PERSON"]
    org_entities = [ent for ent, label in entities if label == "ORG"]

    all_entities = person_entities + org_entities

    for i, entity in enumerate(all_entities):
        G.add_node(entity)
        for j in range(i + 1, len(all_entities)):
            if abs(all_entities.index(entity) - all_entities.index(all_entities[j])) <= 5:
                G.add_edge(entity, all_entities[j])

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=8, font_weight='bold')
    plt.title("Stakeholder Relationships")
    return plt


def generate_timeline(dates):
    """Generates a timeline of events."""
    df = pd.DataFrame(dates, columns=['Date', 'Event'])
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values('Date')

    fig = go.Figure(data=[go.Scatter(
        x=df['Date'],
        y=df['Event'],
        mode='markers+text',
        text=df['Event'],
        textposition='top center'
    )])
    fig.update_layout(title='Case Timeline', xaxis_title='Date', yaxis_title='Event')
    return fig


def generate_entity_summary(entities):
    """Generates a summary of entity types."""
    entity_counts = Counter([ent[1] for ent in entities])
    fig, ax = plt.subplots()
    ax.bar(entity_counts.keys(), entity_counts.values())
    ax.set_title("Entity Types Summary")
    ax.set_xlabel("Entity Type")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
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
<h3><i class="fa fa-gavel"></i> Ask a Question</h3>
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

            # Extract entities and dates
            entities = extract_entities(raw_text)
            dates = extract_dates(raw_text)

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

            # Visualization section in sidebar
            st.sidebar.header("Visualizations")

            visualization_options = st.sidebar.multiselect(
                "Select visualizations to generate",
                ["Stakeholder Graph", "Timeline", "Entity Summary"]
            )

            if "Stakeholder Graph" in visualization_options:
                st.sidebar.subheader("Stakeholder Graph")
                graph = generate_stakeholder_graph(entities)
                st.sidebar.pyplot(graph)

            if "Timeline" in visualization_options:
                st.sidebar.subheader("Case Timeline")
                timeline = generate_timeline(dates)
                st.sidebar.plotly_chart(timeline)

            if "Entity Summary" in visualization_options:
                st.sidebar.subheader("Entity Summary")
                entity_summary = generate_entity_summary(entities)
                st.sidebar.pyplot(entity_summary)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

else:
    st.info("Please upload a case file to begin.")
