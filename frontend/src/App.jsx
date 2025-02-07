import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import axios from 'axios'
import { MagnifyingGlassIcon } from '@heroicons/react/24/outline'
import { motion } from 'framer-motion'
import '@fontsource/open-sans'
import '@fontsource/roboto'
import '@fontsource/lato'

const sampleQueries = [
  "What is the recommended bolus dose of IV nitroglycerin for SCAPE?",
  "What is the available evidence across any trials including but not limited to randomized, observational, and case reports in treating patients with Avycaz and aztreonam for Stenotrophomonas maltophilia infections?",
  "In which subgroup of neurocritically ill patients is stress ulcer prophylaxis recommended?",
  "What safety and efficacy evidence is available for use of fluorescein-benoxinate?",
  "What evidence describes the use of intralesional 5-fluorouracil for skin cancers?"
]

function App() {
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [response, setResponse] = useState({
    overview: "",
    analysis: "",
    recommendations: "",
    key_takeaways: [],
    citations: []
  })
  const [error, setError] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResponse("")
    
    try {
      const apiResponse = await fetch('http://127.0.0.1:5001/api/webSearch', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });
      
      if (!apiResponse.ok) {
        const errorData = await apiResponse.json();
        throw new Error(errorData.error || 'An error occurred');
      }
      
      const data = await apiResponse.json();
      console.log('Received data:', data); // Add logging
      setResponse(data);
      setLoading(false);
    } catch (err) {
      setError('An error occurred while fetching results. Please try again.')
      console.error(err)
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-teal-50 via-blue-50 to-cyan-50 font-['Open_Sans']">
      {/* Background Pattern */}
      <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAiIGhlaWdodD0iNDAiIHZpZXdCb3g9IjAgMCA0MCA0MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxjaXJjbGUgY3g9IjIwIiBjeT0iMjAiIHI9IjIwIiBmaWxsPSIjMDAwIiBmaWxsLW9wYWNpdHk9Ii4wNSIvPjwvZz48L3N2Zz4=')] opacity-10"></div>

      <div className="relative w-full px-4 py-12">
        <div className="max-w-6xl mx-auto">
          {/* Header Section */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-16"
          >
            <h1 className="text-5xl font-bold font-['Lato'] text-transparent bg-clip-text bg-gradient-to-r from-teal-600 to-cyan-600 mb-4">
              Medical Research Assistant
            </h1>
            <p className="text-lg text-gray-600 font-['Roboto']">
              Your AI-powered companion for evidence-based clinical insights
            </p>
          </motion.div>
          
          {/* Search Section */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="max-w-3xl mx-auto mb-12"
          >
            <form onSubmit={handleSubmit} className="mb-4">
              <div className="flex gap-3 p-2 bg-white/80 backdrop-blur-sm rounded-2xl shadow-lg">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Enter your medical research question..."
                  className="flex-1 p-4 text-gray-700 bg-transparent border-2 border-teal-100 rounded-xl focus:ring-2 focus:ring-teal-400 focus:border-teal-400 transition-all duration-200 font-medium placeholder:text-gray-400"
                />
                <button
                  type="submit"
                  disabled={loading || !query.trim()}
                  className="px-8 py-4 bg-gradient-to-r from-teal-500 to-cyan-500 text-white font-semibold rounded-xl shadow-md hover:from-teal-600 hover:to-cyan-600 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:ring-offset-2 disabled:opacity-50 transition-all duration-200 flex items-center gap-2"
                >
                  {loading ? (
                    <span className="flex items-center gap-2">
                      <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
                      </svg>
                      Searching...
                    </span>
                  ) : (
                    <>
                      <MagnifyingGlassIcon className="h-5 w-5" />
                      Search
                    </>
                  )}
                </button>
              </div>
            </form>

            {/* Sample Queries */}
            <div className="text-center text-sm text-gray-500">
              <p className="mb-2">Try asking:</p>
              <div className="flex flex-wrap gap-2 justify-center">
                {sampleQueries.map((sampleQuery, index) => (
                  <button
                    key={index}
                    onClick={() => setQuery(sampleQuery)}
                    className="px-3 py-1 bg-white/50 hover:bg-white/80 rounded-full text-gray-600 text-sm transition-all duration-200"
                  >
                    {sampleQuery}
                  </button>
                ))}
              </div>
            </div>
          </motion.div>

          {/* Error Message */}
          {error && (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="max-w-3xl mx-auto p-4 mb-8 text-red-700 bg-red-50 rounded-xl border border-red-100"
            >
              {error}
            </motion.div>
          )}

          {/* Results Section */}
          {(response.overview || response.analysis) && (
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="max-w-4xl mx-auto bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl p-8 border-2 border-teal-50"
            >
              <div className="prose prose-lg max-w-none prose-headings:text-gray-900 prose-headings:font-bold prose-p:text-gray-700 prose-a:text-teal-600">
                <h2 className="text-2xl font-bold mb-4">Overview</h2>
                <ReactMarkdown>{response.overview}</ReactMarkdown>

                <h2 className="text-2xl font-bold mt-8 mb-4">Analysis</h2>
                <ReactMarkdown>{response.analysis}</ReactMarkdown>

                <h2 className="text-2xl font-bold mt-8 mb-4">Recommendations</h2>
                <ReactMarkdown>{response.recommendations}</ReactMarkdown>

                <h2 className="text-2xl font-bold mt-8 mb-4">Key Takeaways</h2>
                <ul className="list-disc pl-6">
                  {response.key_takeaways.map((point, index) => (
                    <li key={index}><ReactMarkdown>{point}</ReactMarkdown></li>
                  ))}
                </ul>

                <h2 className="text-2xl font-bold mt-8 mb-4">References</h2>
                <ol className="list-decimal pl-6">
                  {response.citations.map((citation, index) => (
                    <li key={index} className="mb-2">
                      {citation.url ? (
                        <a 
                          href={citation.url} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="text-teal-600 hover:text-teal-800 hover:underline"
                        >
                          {citation.reference}
                        </a>
                      ) : (
                        <span>{citation.reference}</span>
                      )}
                    </li>
                  ))}
                </ol>
              </div>
            </motion.div>
          )}
        </div>
      </div>

      {/* Footer */}
      <footer className="w-full py-8 mt-16 bg-white/30 backdrop-blur-sm border-t border-teal-100">
        <div className="max-w-6xl mx-auto px-4">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <div className="text-sm text-gray-600">
              2025 Medical Research Assistant
            </div>
            <div className="flex items-center gap-6 text-sm">
              <a href="mailto:aniketmaj21@gmail.com" className="text-gray-600 hover:text-teal-600 transition-colors duration-200 flex items-center gap-2">
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M2.003 5.884L10 9.882l7.997-3.998A2 2 0 0016 4H4a2 2 0 00-1.997 1.884z"></path>
                  <path d="M18 8.118l-8 4-8-4V14a2 2 0 002 2h12a2 2 0 002-2V8.118z"></path>
                </svg>
                aniketmaj21@gmail.com
              </a>
              <a href="https://github.com/amajety1" target="_blank" rel="noopener noreferrer" className="text-gray-600 hover:text-teal-600 transition-colors duration-200 flex items-center gap-2">
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                </svg>
                github.com/amajety1
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App
