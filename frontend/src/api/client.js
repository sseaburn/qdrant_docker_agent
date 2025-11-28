import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const api = axios.create({
  baseURL: API_URL,
})

export const uploadDocument = (file) => {
  const formData = new FormData()
  formData.append('file', file)
  return api.post('/api/documents/upload', formData)
}

export const getDocuments = () => api.get('/api/documents')

export const getDocument = (id) => api.get(`/api/documents/${id}`)

export const deleteDocument = (id) => api.delete(`/api/documents/${id}`)

export const sendMessage = (message) => api.post('/api/chat', { message })

export const healthCheck = () => api.get('/health')
