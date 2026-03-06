import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv

# LangChain Imports — compatible with langchain 1.x (no deprecated RetrievalQA)
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. SETUP & CONFIGURATION
load_dotenv(find_dotenv())
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if not HUGGINGFACE_API_KEY:
    raise ValueError("HUGGINGFACE_API_KEY not found in environment variables.")

DATASET_PATH = r"C:\Users\Lenovo\Downloads\archive (3)\FashionDataset.csv"

# ─────────────────────────────────────────────────────────────
# ROUTER — decides if a query needs product catalogue search
# (→ RAG chain) or can be answered directly (→ plain LLM)
#pip install --upgrade langchain langchain-core langchain-community langchain-huggingface langchain-text-splitters faiss-cpu sentence-transformers
# ─────────────────────────────────────────────────────────────
PRODUCT_KEYWORDS = {
    # clothing items
    "dress", "shirt", "jeans", "shoes", "kurta", "saree", "jacket",
    "top", "skirt", "leggings", "trouser", "pant", "kurti", "suit",
    "blazer", "hoodie", "sweater", "shorts", "accessories",
    # shopping intent
    "brand", "category", "product", "item", "cloth", "clothing",
    "show me", "find me", "do you have", "available", "stock",
    "buy", "purchase", "collection", "latest", "new arrival",
    # price / discount
    "price", "cost", "how much", "mrp", "selling price",
    "cheap", "expensive", "discount", "offer", "sale", "deal",
    # style / recommendation
    "recommend", "suggest", "suggestion", "best",
    "size", "colour", "color", "fit",
}

def is_product_query(query: str) -> bool:
    """Returns True when the query is about products, prices, or catalogue."""
    q = query.lower()
    return any(kw in q for kw in PRODUCT_KEYWORDS)


def run_rag_bot():
    print("--- Starting E-commerce RAG Bot Pipeline ---")

    # 2. LOAD & PREPROCESS DATA
    print("\n[Step 1/5] Loading and cleaning dataset...")
    df = pd.read_csv(DATASET_PATH)

    # Sample 1000 products for quick demo; raise limit for larger coverage
    df_sample = df.sample(n=min(1000, len(df)), random_state=42).fillna("N/A")

    # Convert each row into a descriptive string (Document)
    documents = []
    for _, row in df_sample.iterrows():
        content = (
            f"Brand: {row['BrandName']}. "
            f"Category: {row['Category']}. "
            f"Product: {row['Deatils']}. "
            f"Available Sizes: {row['Sizes']}. "
            f"MRP: {row['MRP']}. "
            f"Selling Price: {row['SellPrice']}. "
            f"Discount: {row['Discount']}."
        )
        doc = Document(
            page_content=content,
            metadata={"brand": row['BrandName'], "category": row['Category']}
        )
        documents.append(doc)

    # 3. EMBEDDINGS & VECTOR DATABASE
    print("\n[Step 2/5] Creating local Vector Database (Indexing)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(documents, embeddings)
    print("Vector Database initialized successfully.")

    # 4. LLM SETUP
    print("\n[Step 3/5] Initializing LLM (Hugging Face API)...")
    endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        huggingfacehub_api_token=HUGGINGFACE_API_KEY,
        temperature=0.2,
        max_new_tokens=512,
    )
    llm = ChatHuggingFace(llm=endpoint)

    # 5. RAG CHAIN — answers product / price / catalogue questions
    print("\n[Step 4/5] Setting up RAG + General chains...")

    rag_prompt = PromptTemplate(
        template="""You are a helpful e-commerce shopping assistant.
Use the following product catalogue context to answer the user's question.
If the exact product isn't listed, suggest the closest available option.
Never make up product details — if unsure, say so.

Context:
{context}

Question: {question}

Assistant's Response:""",
        input_variables=["context", "question"],
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # 6. GENERAL LLM CHAIN — answers everything else
    #    (price tips, fashion advice, brand comparisons, shopping guidance …)
    general_chain = (
        ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a knowledgeable e-commerce and fashion assistant. "
                "Answer general shopping questions — pricing advice, fashion tips, "
                "brand comparisons, style guides, size guides, and anything a "
                "shopper might ask. Be concise, friendly, and accurate.",
            ),
            ("human", "{question}"),
        ])
        | llm
        | StrOutputParser()
    )

    # 7. SMART ROUTER — picks the right chain per query
    def smart_router(query: str) -> str:
        if is_product_query(query):
            print("  [Router] Product/Price query → RAG chain 🔍")
            return rag_chain.invoke(query)
        else:
            print("  [Router] General query → LLM chain 💬")
            return general_chain.invoke({"question": query})

    # 8. INTERACTIVE CHAT
    print("\n[Step 5/5] Bot Ready!")
    print("\nAsk me anything — product searches, prices, discounts, fashion tips & more.")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        print("Thinking...")
        try:
            result = smart_router(query)
            print(f"\nBot: {result}\n")
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    run_rag_bot()
