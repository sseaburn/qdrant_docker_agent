import { FileText, Trash2 } from 'lucide-react'

function FileList({ documents = [], onDelete }) {
  if (documents.length === 0) {
    return (
      <div className="text-center py-8 text-gray-400">
        <p>No documents uploaded yet</p>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      {documents.map((doc) => (
        <div
          key={doc.id}
          className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
        >
          <div className="flex items-center gap-2">
            <FileText className="w-5 h-5 text-blue-500" />
            <span className="text-sm font-medium">{doc.filename}</span>
          </div>
          <button
            onClick={() => onDelete(doc.id)}
            className="p-1 text-gray-400 hover:text-red-500 transition-colors"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      ))}
    </div>
  )
}

export default FileList
