

# **SherlockSearch**

SherlockSearch is a clinical AI assistant designed to retrieve, summarize, and analyze medical literature. It leverages web scraping, query expansion, and GPT-4o to generate structured, evidence-based summaries tailored for healthcare professionals.

## **Getting Started**

Follow the instructions below to set up and run the application.

---

### **1. Clone the Repository**
```bash
git clone https://github.com/sid-inpharmd/sherlockSearch.git
cd sherlockSearch
```

---

### **2. Install Dependencies**

#### Backend Dependencies
Ensure you have Python 3.8 or higher installed.

Install the required Python libraries using `pip`:
```bash
pip install -r requirements.txt
```

#### Frontend Dependencies
Ensure you have Node.js and npm installed.

Navigate to the frontend directory and install dependencies:
```bash
cd frontend
npm install
```

---

### **3. Run the Application**

#### Start the Backend API
Start the Flask API by executing the following command:
```bash
python api.py
```

This will start the server, and it should listen on `http://127.0.0.1:5000`.

#### Start the Frontend Development Server
In a new terminal, navigate to the frontend directory and start the development server:
```bash
cd frontend
npm run dev
```

This will start the frontend development server, typically accessible at `http://localhost:3000`.

---

### **4. Test the API**
You can send a POST request using `curl`:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"query": "How should asthma management change in pediatric patients during viral infections?"}' http://127.0.0.1:5000/api/webSearch
```

---

## **Project Structure**

### **Key Files**
1. **`api.py`**
   - **Purpose**: Main application file that defines the Flask API.
   - **Key Functions**:
     - **`queryExpansion`**: Expands user queries to generate a list of related questions.
     - **`rank_results_with_chunking`**: Ranks search results based on similarity to the query.
     - **`summarizer`**: Summarizes the combined scraped content for clinical relevance.
     - **`sherlockSearch`**: Main route to handle POST requests, process queries, and return a structured response.

2. **`search.py`**
   - **Purpose**: Handles web scraping and content summarization.
   - **Key Functions**:
     - **`search_api`**: Queries web search engines to retrieve relevant results.
     - **`filter_process`**: Filters and processes web scraping results for PubMed and Springer.
     - **`run`**: Executes multiple search queries concurrently.
     - **`summarize_scraped_content`**: Summarizes individual scraped results using GPT-4o.

3. **`requirements.txt`**
   - Contains the list of dependencies required to run the project:
     - Flask (for API handling)
     - Flask-CORS (for Cross-Origin Resource Sharing)
     - SentenceTransformers (for query similarity ranking)
     - Requests (for web scraping)
     - BeautifulSoup (for HTML parsing)
     - OpenAI Python SDK (for GPT-4o integration)

4. **`conf/params.yml`**
   - Configuration file to store additional parameters for web scraping or API calls.

5. **`data/`**
   - **Purpose**: Stores the output files generated by the application.
   - Example Files:
     - `summarized_content.txt`: Summaries of the scraped content.
     - `final_response.txt`: Final summarized response sent to the user.

---

## **Features**
- **Query Expansion**: Automatically generates related queries for comprehensive search results.
- **Web Scraping**: Extracts content from PubMed, Springer, and other medical literature sources.
- **Summarization**: Produces concise and clinically relevant summaries of the scraped content.
- **Ranking**: Identifies the most relevant content using Sentence Transformers.

---

## **Notes**
- Ensure you have valid API keys for external services such as SerpAPI and OpenAI.
- Adjust `search.py` to include additional filtering logic for other medical literature platforms if needed.

---

