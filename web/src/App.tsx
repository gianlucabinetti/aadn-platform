import { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Login from './components/Login'
import Dashboard from './pages/Dashboard'
import Decoys from './pages/Decoys'
import Monitoring from './pages/Monitoring'
import Intelligence from './pages/Intelligence'
import Settings from './pages/Settings'

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Check for existing authentication
    const token = localStorage.getItem('aadn_token')
    const userData = localStorage.getItem('aadn_user')
    
    if (token && userData) {
      // Verify token is still valid
      fetch('/api/auth/me', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
      .then(response => {
        if (response.ok) {
          setIsAuthenticated(true)
          setUser(JSON.parse(userData))
        } else {
          // Token is invalid, clear storage
          localStorage.removeItem('aadn_token')
          localStorage.removeItem('aadn_user')
        }
      })
      .catch(() => {
        // Network error, clear storage
        localStorage.removeItem('aadn_token')
        localStorage.removeItem('aadn_user')
      })
      .finally(() => {
        setLoading(false)
      })
    } else {
      setLoading(false)
    }
  }, [])

  const handleLogin = (_token: string, userData: any) => {
    setIsAuthenticated(true)
    setUser(userData)
  }

  const handleLogout = () => {
    const token = localStorage.getItem('aadn_token')
    
    if (token) {
      fetch('/api/auth/logout', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
    }
    
    localStorage.removeItem('aadn_token')
    localStorage.removeItem('aadn_user')
    setIsAuthenticated(false)
    setUser(null)
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-white text-xl">Loading...</div>
      </div>
    )
  }

  if (!isAuthenticated) {
    return <Login onLogin={handleLogin} />
  }

  return (
    <Router>
      <Layout user={user} onLogout={handleLogout}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/decoys" element={<Decoys />} />
          <Route path="/monitoring" element={<Monitoring />} />
          <Route path="/intelligence" element={<Intelligence />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </Layout>
    </Router>
  )
}

export default App 