import { useState, useEffect } from 'react'
import { EyeIcon, ClockIcon, GlobeAltIcon } from '@heroicons/react/24/outline'

interface Interaction {
  id: string
  timestamp: string
  sourceIp: string
  decoyName: string
  interactionType: string
  details: string
  severity: 'low' | 'medium' | 'high'
}

export default function Monitoring() {
  const [interactions, setInteractions] = useState<Interaction[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchInteractions = async () => {
      try {
        await new Promise(resolve => setTimeout(resolve, 1000))
        
        setInteractions([
          {
            id: '1',
            timestamp: '2024-01-15T14:30:00Z',
            sourceIp: '192.168.1.100',
            decoyName: 'SSH Honeypot',
            interactionType: 'ssh_connection',
            details: 'SSH connection attempt with username: admin',
            severity: 'medium'
          },
          {
            id: '2',
            timestamp: '2024-01-15T14:25:00Z',
            sourceIp: '10.0.0.50',
            decoyName: 'HTTP Web Server',
            interactionType: 'http_request',
            details: 'GET /admin/login.php',
            severity: 'low'
          },
          {
            id: '3',
            timestamp: '2024-01-15T14:20:00Z',
            sourceIp: '172.16.0.25',
            decoyName: 'FTP Server',
            interactionType: 'ftp_login_attempt',
            details: 'Failed login attempt: anonymous/password123',
            severity: 'high'
          }
        ])
      } catch (error) {
        console.error('Failed to fetch interactions:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchInteractions()
  }, [])

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high':
        return 'text-danger-500'
      case 'medium':
        return 'text-warning-500'
      case 'low':
        return 'text-success-500'
      default:
        return 'text-gray-500'
    }
  }

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="loading-shimmer h-8 w-32 rounded"></div>
        <div className="space-y-4">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="card p-6">
              <div className="loading-shimmer h-6 w-full rounded"></div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold text-white">Monitoring</h2>
        <p className="mt-2 text-gray-400">
          Real-time monitoring of decoy interactions and security events
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-3">
        <div className="card p-6">
          <div className="flex items-center">
                            <div className="flex-shrink-0 p-3 rounded-lg bg-blue-500/10">
                  <EyeIcon className="h-6 w-6 text-blue-500" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-400">Total Interactions</p>
              <p className="text-2xl font-semibold text-white">127</p>
            </div>
          </div>
        </div>
        <div className="card p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0 p-3 rounded-lg bg-warning-500/10">
              <ClockIcon className="h-6 w-6 text-warning-500" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-400">Last 24h</p>
              <p className="text-2xl font-semibold text-white">45</p>
            </div>
          </div>
        </div>
        <div className="card p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0 p-3 rounded-lg bg-success-500/10">
              <GlobeAltIcon className="h-6 w-6 text-success-500" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-400">Unique IPs</p>
              <p className="text-2xl font-semibold text-white">23</p>
            </div>
          </div>
        </div>
      </div>

      {/* Interactions Table */}
      <div className="card overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-700">
          <h3 className="text-lg font-medium text-white">Recent Interactions</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-700">
                            <thead className="bg-gray-900">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Timestamp
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Source IP
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Decoy
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Type
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Details
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Severity
                </th>
              </tr>
            </thead>
            <tbody className="bg-gray-900 divide-y divide-gray-700">
              {interactions.map((interaction) => (
                                  <tr key={interaction.id} className="hover:bg-gray-700">
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                    {new Date(interaction.timestamp).toLocaleString()}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-white">
                    {interaction.sourceIp}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-white">
                    {interaction.decoyName}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-500/10 text-blue-400">
                      {interaction.interactionType}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-300 max-w-xs truncate">
                    {interaction.details}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`text-sm font-medium ${getSeverityColor(interaction.severity)}`}>
                      {interaction.severity.toUpperCase()}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
} 