import { useState } from 'react'
import { Send, AlertCircle } from 'lucide-react'
import MessageBubble from './MessageBubble'
import { sendMessage } from '../api/client'

function ChatInterface({ hasDocuments = false }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage = { role: 'user', content: input }
    setMessages((prev) => [...prev, userMessage])
    setInput('')
    setIsLoading(true)
    setError(null)

    try {
      const response = await sendMessage(input)
      const { answer, sources } = response.data

      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: answer,
          sources: sources || [],
        },
      ])
    } catch (err) {
      console.error('Chat error:', err)
      setError(err.response?.data?.detail || 'Failed to get response')
      // Remove the user message on error
      setMessages((prev) => prev.slice(0, -1))
    } finally {
      setIsLoading(false)
    }
  }

  const clearChat = () => {
    setMessages([])
    setError(null)
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header with clear button */}
      {messages.length > 0 && (
        <div className="flex justify-end mb-2">
          <button
            onClick={clearChat}
            className="text-xs text-gray-400 hover:text-gray-600 transition-colors"
          >
            Clear chat
          </button>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto space-y-4 mb-4">
        {messages.length === 0 ? (
          <div className="text-center text-gray-400 py-8">
            {hasDocuments ? (
              <>
                <p className="text-lg">Ask a question about your documents</p>
                <p className="text-sm mt-1">
                  I'll search through your uploaded PDFs to find answers
                </p>
              </>
            ) : (
              <>
                <p className="text-lg">Upload documents to get started</p>
                <p className="text-sm mt-1">
                  Upload PDF files, then ask questions about their content
                </p>
              </>
            )}
          </div>
        ) : (
          messages.map((msg, idx) => (
            <MessageBubble key={idx} message={msg} />
          ))
        )}
        {isLoading && (
          <div className="flex items-center gap-2 text-gray-400">
            <div className="flex space-x-1">
              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
            </div>
            <span>Thinking...</span>
          </div>
        )}
      </div>

      {/* Error message */}
      {error && (
        <div className="flex items-center gap-2 p-2 mb-2 bg-red-50 text-red-600 rounded-lg text-sm">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Input */}
      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={hasDocuments ? "Ask a question..." : "Upload documents first..."}
          className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-50"
          disabled={isLoading}
        />
        <button
          type="submit"
          disabled={isLoading || !input.trim()}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          title="Send message"
        >
          <Send className="w-5 h-5" />
        </button>
      </form>
    </div>
  )
}

export default ChatInterface
