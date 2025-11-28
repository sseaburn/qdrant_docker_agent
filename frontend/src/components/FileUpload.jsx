import { Upload } from 'lucide-react'

function FileUpload({ onUpload }) {
  return (
    <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-blue-400 transition-colors">
      <Upload className="w-8 h-8 mx-auto mb-2 text-gray-400" />
      <p className="text-gray-500">Drag and drop PDF files here</p>
      <p className="text-sm text-gray-400 mt-1">or click to browse</p>
    </div>
  )
}

export default FileUpload
