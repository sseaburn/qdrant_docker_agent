import { useState, useRef } from 'react'
import { Upload, AlertCircle, CheckCircle, Loader } from 'lucide-react'

function FileUpload({ onUpload, isUploading = false }) {
  const [isDragging, setIsDragging] = useState(false)
  const [error, setError] = useState(null)
  const fileInputRef = useRef(null)

  const validateFile = (file) => {
    if (!file) return 'No file selected'
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      return 'Only PDF files are allowed'
    }
    if (file.size > 50 * 1024 * 1024) {
      return 'File size must be less than 50MB'
    }
    return null
  }

  const handleFile = async (file) => {
    const validationError = validateFile(file)
    if (validationError) {
      setError(validationError)
      return
    }

    setError(null)
    try {
      await onUpload(file)
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to upload file')
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    handleFile(file)
  }

  const handleClick = () => {
    fileInputRef.current?.click()
  }

  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      handleFile(file)
    }
    // Reset input so same file can be selected again
    e.target.value = ''
  }

  return (
    <div className="space-y-2">
      <div
        onClick={handleClick}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
          isDragging
            ? 'border-blue-500 bg-blue-50'
            : isUploading
            ? 'border-gray-300 bg-gray-50 cursor-wait'
            : 'border-gray-300 hover:border-blue-400'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          onChange={handleFileChange}
          className="hidden"
          disabled={isUploading}
        />

        {isUploading ? (
          <>
            <Loader className="w-8 h-8 mx-auto mb-2 text-blue-500 animate-spin" />
            <p className="text-blue-600 font-medium">Uploading...</p>
            <p className="text-sm text-gray-400 mt-1">Processing PDF</p>
          </>
        ) : (
          <>
            <Upload className="w-8 h-8 mx-auto mb-2 text-gray-400" />
            <p className="text-gray-500">Drag and drop PDF files here</p>
            <p className="text-sm text-gray-400 mt-1">or click to browse</p>
          </>
        )}
      </div>

      {error && (
        <div className="flex items-center gap-2 p-2 bg-red-50 text-red-600 rounded-lg text-sm">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}
    </div>
  )
}

export default FileUpload
