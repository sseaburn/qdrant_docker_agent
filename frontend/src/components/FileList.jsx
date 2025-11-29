import { FileText, Trash2, Loader, CheckCircle, AlertCircle } from 'lucide-react'

function FileList({ documents = [], onDelete, isDeleting = null }) {
  if (documents.length === 0) {
    return (
      <div className="text-center py-8 text-gray-400">
        <FileText className="w-12 h-12 mx-auto mb-2 opacity-50" />
        <p>No documents uploaded yet</p>
        <p className="text-sm mt-1">Upload a PDF to get started</p>
      </div>
    )
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'processing':
        return <Loader className="w-4 h-4 text-yellow-500 animate-spin" />
      case 'ready':
        return <CheckCircle className="w-4 h-4 text-green-500" />
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-500" />
      default:
        return null
    }
  }

  const getStatusText = (status) => {
    switch (status) {
      case 'processing':
        return 'Processing...'
      case 'ready':
        return 'Ready'
      case 'error':
        return 'Error'
      default:
        return status
    }
  }

  return (
    <div className="space-y-2">
      {documents.map((doc) => (
        <div
          key={doc.id}
          className={`flex items-center justify-between p-3 rounded-lg transition-colors ${
            isDeleting === doc.id
              ? 'bg-red-50 opacity-50'
              : 'bg-gray-50 hover:bg-gray-100'
          }`}
        >
          <div className="flex items-center gap-3 min-w-0 flex-1">
            <FileText className="w-5 h-5 text-blue-500 flex-shrink-0" />
            <div className="min-w-0 flex-1">
              <p className="text-sm font-medium truncate" title={doc.filename}>
                {doc.filename}
              </p>
              <div className="flex items-center gap-2 text-xs text-gray-500">
                <span className="flex items-center gap-1">
                  {getStatusIcon(doc.status)}
                  {getStatusText(doc.status)}
                </span>
                {doc.chunk_count > 0 && (
                  <span className="text-gray-400">
                    • {doc.chunk_count} chunks
                  </span>
                )}
              </div>
            </div>
          </div>
          <button
            onClick={() => onDelete(doc.id)}
            disabled={isDeleting === doc.id}
            className="p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            title="Delete document"
          >
            {isDeleting === doc.id ? (
              <Loader className="w-4 h-4 animate-spin" />
            ) : (
              <Trash2 className="w-4 h-4" />
            )}
          </button>
        </div>
      ))}
    </div>
  )
}

export default FileList
