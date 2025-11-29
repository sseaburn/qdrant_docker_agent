import { useState, useEffect, useCallback } from 'react'
import { FileText, MessageSquare, RefreshCw } from 'lucide-react'
import FileUpload from './components/FileUpload'
import FileList from './components/FileList'
import ChatInterface from './components/ChatInterface'
import { uploadDocument, getDocuments, deleteDocument, healthCheck } from './api/client'

function App() {
  const [status, setStatus] = useState('checking...')
  const [documents, setDocuments] = useState([])
  const [isUploading, setIsUploading] = useState(false)
  const [isDeleting, setIsDeleting] = useState(null)
  const [isLoading, setIsLoading] = useState(true)

  // Check backend health
  useEffect(() => {
    const checkHealth = async () => {
      try {
        await healthCheck()
        setStatus('connected')
      } catch (error) {
        setStatus('disconnected')
      }
    }
    checkHealth()
    // Check health every 30 seconds
    const interval = setInterval(checkHealth, 30000)
    return () => clearInterval(interval)
  }, [])

  // Load documents on mount
  const loadDocuments = useCallback(async () => {
    setIsLoading(true)
    try {
      const response = await getDocuments()
      setDocuments(response.data.documents || [])
    } catch (error) {
      console.error('Failed to load documents:', error)
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    loadDocuments()
  }, [loadDocuments])

  // Handle file upload
  const handleUpload = async (file) => {
    setIsUploading(true)
    try {
      const response = await uploadDocument(file)
      setDocuments((prev) => [...prev, response.data])
    } finally {
      setIsUploading(false)
    }
  }

  // Handle document deletion
  const handleDelete = async (docId) => {
    setIsDeleting(docId)
    try {
      await deleteDocument(docId)
      setDocuments((prev) => prev.filter((doc) => doc.id !== docId))
    } catch (error) {
      console.error('Failed to delete document:', error)
    } finally {
      setIsDeleting(null)
    }
  }

  return (
    <div className="flex h-screen bg-gray-100">
      {/* Left Panel - Documents */}
      <div className="w-1/3 border-r border-gray-200 bg-white flex flex-col">
        {/* Header */}
        <div className="p-4 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <FileText className="w-6 h-6 text-blue-600" />
              <h2 className="text-xl font-bold text-gray-800">Documents</h2>
            </div>
            <button
              onClick={loadDocuments}
              disabled={isLoading}
              className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded transition-colors disabled:opacity-50"
              title="Refresh documents"
            >
              <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>

        {/* Upload area */}
        <div className="p-4 border-b border-gray-200">
          <FileUpload onUpload={handleUpload} isUploading={isUploading} />
        </div>

        {/* Document list */}
        <div className="flex-1 overflow-y-auto p-4">
          <FileList
            documents={documents}
            onDelete={handleDelete}
            isDeleting={isDeleting}
          />
        </div>

        {/* Status indicator */}
        <div className="p-3 border-t border-gray-200 bg-gray-50">
          <p className="text-xs text-gray-500">
            Backend:{' '}
            <span
              className={`font-medium ${
                status === 'connected'
                  ? 'text-green-600'
                  : status === 'checking...'
                  ? 'text-yellow-600'
                  : 'text-red-600'
              }`}
            >
              {status}
            </span>
            {documents.length > 0 && (
              <span className="ml-2 text-gray-400">
                • {documents.length} document{documents.length !== 1 ? 's' : ''}
              </span>
            )}
          </p>
        </div>
      </div>

      {/* Right Panel - Chat */}
      <div className="w-2/3 flex flex-col">
        {/* Header */}
        <div className="p-4 border-b border-gray-200 bg-white">
          <div className="flex items-center gap-2">
            <MessageSquare className="w-6 h-6 text-green-600" />
            <h2 className="text-xl font-bold text-gray-800">Chat</h2>
          </div>
        </div>

        {/* Chat interface */}
        <div className="flex-1 p-4 overflow-hidden">
          <ChatInterface hasDocuments={documents.length > 0} />
        </div>
      </div>
    </div>
  )
}

export default App
