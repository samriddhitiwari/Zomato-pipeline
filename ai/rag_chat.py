import os
import numpy as np
import pandas as pd
import streamlit as st
import snowflake.connector
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
NEW_REVIEWS = 500
TOK_K = 5
CACHE_FILE = "review_embeddings.parquet"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def read_reviews_from_snowflake():
    conn = snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
    )

    query = f"""
        SELECT REVIEW_ID, CITY, RATING, COMMENT
        FROM ZOMATO.STAGING.STG_REVIEWS
        SAMPLE ({NEW_REVIEWS} ROWS)
    """
    df = conn.cursor().execute(query).fetch_pandas_all()
    conn.close()

    df.columns = [col.lower() for col in df.columns]
    return df

def embed(texts):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts)

    return [item.embedding for item in response.data]
    
@st.cache_data()
def load_reviews():
    if os.path.exists(CACHE_FILE):
        return pd.read_parquet(CACHE_FILE)
    #instead of going through the AI model agin and again we will cache the embeddings in a parquet file, this also saves cost
    #if the file exists we will read it from there, if not we will read it from snowflake and then embed it and save it to parquet file

    df = read_reviews_from_snowflake()
    df['embedding'] = embed(df['comment'].tolist())
    df.to_parquet(CACHE_FILE)
    return df
#we get the reviews from snowflake, we embed them and save them, 
# if the file exists we will read it from there, if not we will read it from snowflake and then embed it and save it
#we want to convert text into embeddings format, we will use the embedding model to convert the text into embeddings,
#then we cache those files into our local- review embedings we created
#we will use the embeddings to find similar reviews to the question asked by the user

st.title("Chat with your Zomato Reviews")
st.caption(f"Searching {NEW_REVIEWS} review, answering with {CHAT_MODEL} model")

def consine_simiarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

def find_similar_reviews(question, df):
    question_vector = embed([question])[0]
#machine only understands numbers, so when we loaded the reviews,
#we put that into df embedding, now whwen we get the question from the user,
#we will embed that question into embedding model and convert it into a vector,
#then we will compare it with the review embeddings to find the most similar reviews
    scores = []
    for review_vector in df['embedding']:
        scores.append(consine_simiarity(question_vector, review_vector))
#we are checming the similarity between the question and the reviews,
#we will append cosine similarity scores to get a score for each review and question pair,
#we check the siliarity between rewiews and the question that the user has asked 
#once we have the scores for each review, we will sort them and get the top 5 reviews
#then we pass the top 5 reviews to the LLM model to get the answer for the question asked by the user

    df = df.copy()
    df['score'] = scores
    return df.nlargest(TOK_K, 'score')

def ask_llm(question, top_reviews):
    conext = ""

    for _, row in top_reviews.iterrows():
        conext += f" ({row['city']}, {row['rating']} stars) {row['comment']}\n"

    system_prompt = (
        "Answer ONLY using the customer reviews provided. "
        "Be concise. If the reviews don't covert it, say so"
    )

    user_prompt = f"Questions: {question}\n\nReviews:\n{conext}"

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content
    
review_df = load_reviews()

question = st.text_input("Ask a question about your reviews:",
                         placeholder="e.g. What are the most common complaints about delivery?")

if question:
    top_reviews = find_similar_reviews(question, review_df)
    answer = ask_llm(question, top_reviews)

    st.markdown(f"**Answer:**")
    st.write(answer)

    with st.expander("Reviews used to build this answer"):
        st.dataframe(top_reviews[['city', 'rating', 'comment']], hide_index=True)