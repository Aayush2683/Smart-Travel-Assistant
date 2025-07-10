# Smart-Travel-Assistant
**AI-powered toolkit that predicts flight fares and answers travel-document questions in real time.**

The app combines two ML pipelines:

1. **Price-Prediction Engine** – Gradient-boosted trees trained on historic domestic-route data, with a rule-based fuel-proxy adjustment.  
2. **Travel-Document RAG** – Retrieval-Augmented Generation that embeds airline PDFs (baggage rules, fees, dangerous goods, etc.) into a FAISS vector DB and lets GPT answer questions with page-level citations.
