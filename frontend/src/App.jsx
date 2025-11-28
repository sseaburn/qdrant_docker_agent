import { useState, useEffect } from 'react'
import { FileText, MessageSquare } from 'lucide-react'

function App() {
  const [status, setStatus] = useState('checking...')

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
        const response = await fetch(`${apiUrl}/health`)
        if (response.ok) {
          setStatus('connected')
        } else {
          setStatus('error')
        }
      } catch (error) {
        setStatus('disconnected')
      }
    }
    checkHealth()
  }, [])

  return (
    <div className="flex h-screen bg-gray-100">
      {/* Left Panel - Documents */}
      <div className="w-1/3 border-r border-gray-200 bg-white p-4 flex flex-col">
        <div className="flex items-center gap-2 mb-4">
          <FileText className="w-6 h-6 text-blue-600" />
          <h2 className="text-xl font-bold text-gray-800">Documents</h2>
        </div>

        <div className="flex-1 flex items-center justify-center text-gray-400">
          <p>Document upload coming in Phase 2</p>
        </div>
      </div>

      {/* Right Panel - Chat */}
      <div className="w-2/3 p-4 flex flex-col">
        <div className="flex items-center gap-2 mb-4">
          <MessageSquare className="w-6 h-6 text-green-600" />
          <h2 className="text-xl font-bold text-gray-800">Chat</h2>
        </div>

        <div className="flex-1 flex items-center justify-center text-gray-400">
          <p>Chat interface coming in Phase 5</p>
        </div>

        {/* Status indicator */}
        <div className="mt-4 p-3 bg-gray-50 rounded-lg">
          <p className="text-sm text-gray-600">
            Backend status:{' '}
            <span className={`font-medium ${
              status === 'connected' ? 'text-green-600' :
              status === 'checking...' ? 'text-yellow-600' : 'text-red-600'
            }`}>
              {status}
            </span>
          </p>
        </div>
      </div>
    </div>
  )
}

export default App
