import streamlit as st
import json
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Memory Assistant", layout="wide")

FILE = "memory_store.json"

# ---------------- STYLING ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #141E30, #243B55);
}
.main {
    background: rgba(255,255,255,0.05);
    border-radius: 15px;
    padding: 20px;
}
.card {
    background: rgba(255,255,255,0.08);
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 10px;
    transition: 0.3s;
}
.card:hover {
    transform: scale(1.02);
}
.priority-high {color: #ff4b5c;}
.priority-medium {color: #ffa600;}
.priority-low {color: #00c6ff;}
</style>
""", unsafe_allow_html=True)

# ---------------- DATA FUNCTIONS ----------------
def load_memories():
    try:
        with open(FILE, "r") as f:
            return json.load(f)
    except:
        return []

def save_memory(text, category, priority):
    memories = load_memories()

    memory = {
        "text": text,
        "category": category,
        "priority": priority,
        "time": str(datetime.now())
    }

    memories.append(memory)

    with open(FILE, "w") as f:
        json.dump(memories, f, indent=4)

def search_memory(query):
    memories = load_memories()
    if not memories:
        return []

    texts = [m["text"] for m in memories]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts + [query])

    similarity = cosine_similarity(vectors[-1], vectors[:-1])
    scores = list(enumerate(similarity[0]))
    scores.sort(key=lambda x: x[1], reverse=True)

    return [memories[i] for i, _ in scores[:5]]

# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align:center;'> Memory System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Organize your Thoughts </p>", unsafe_allow_html=True)

menu = st.sidebar.radio("Navigation", ["Add Memory", "Search", "Dashboard"])

# ---------------- ADD MEMORY ----------------
if menu == "Add Memory":
    st.subheader("📝 Add New Memory")

    text = st.text_area("Write your memory...")

    col1, col2 = st.columns(2)

    with col1:
        category = st.selectbox("Category", ["Work", "Personal", "Learning"])

    with col2:
        priority = st.selectbox("Priority", ["High", "Medium", "Low"])

    if st.button("Save"):
        if text.strip():
            save_memory(text, category, priority)
            st.success("Memory saved successfully!")
        else:
            st.warning("Write something first!")

# ---------------- SEARCH ----------------
elif menu == "Search":
    st.subheader("🔍 Search Your Memories")

    query = st.text_input("Search anything...")

    if st.button("Find"):
        results = search_memory(query)

        if results:
            for m in results:
                st.markdown(f"""
                <div class="card">
                    <b>{m['text']}</b><br>
                    📂 {m.get('category', 'Uncategorized')}
                    ⚡ <span class="priority-{m['priority'].lower()}">{m['priority']}</span><br>
                    🕒 {m['time']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("No results found")

# ---------------- DASHBOARD ----------------
elif menu == "Dashboard":
    st.subheader("📊 Overview")

    memories = load_memories()

    st.write(f"Total Memories: {len(memories)}")

    categories = {}
    for m in memories:
       cat = m.get("category", "Uncategorized")
       categories[cat] = categories.get(cat, 0) + 1

    st.write("📂 Category Breakdown:")
    st.write(categories)

    st.write("🕒 Recent Memories:")
    for m in memories[-5:]:
        st.markdown(f"""
        <div class="card">
            {m['text']}
        </div>
        """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Built like a real product 💼</p>", unsafe_allow_html=True)