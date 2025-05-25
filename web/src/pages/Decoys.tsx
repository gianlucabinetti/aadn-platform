import { useState, useEffect } from 'react'
import {
  PlusIcon,
  PlayIcon,
  StopIcon,
  TrashIcon,
  EyeIcon,
  Cog6ToothIcon,
} from '@heroicons/react/24/outline'
import clsx from 'clsx'

interface Decoy {
  id: string
  name: string
  type: string
  status: 'active' | 'inactive' | 'error' | 'pending'
  host: string
  port: number
  interactions: number
  lastActivity: string
  created_at: string
}

export default function Decoys() {
  const [decoys, setDecoys] = useState<Decoy[]>([])
  const [loading, setLoading] = useState(true)
  const [showCreateModal, setShowCreateModal] = useState(false)

  useEffect(() => {
    // Simulate API call
    const fetchDecoys = async () => {
      try {
        await new Promise(resolve => setTimeout(resolve, 1000))
        
        // Sample data
        setDecoys([
          {
            id: '1',
            name: 'SSH Honeypot',
            type: 'ssh',
            status: 'active',
            host: '127.0.0.1',
            port: 2222,
            interactions: 45,
            lastActivity: '2 minutes ago',
            created_at: '2024-01-15T10:30:00Z'
          },
          {
            id: '2',
            name: 'HTTP Web Server',
            type: 'http',
            status: 'active',
            host: '127.0.0.1',
            port: 8080,
            interactions: 23,
            lastActivity: '5 minutes ago',
            created_at: '2024-01-15T09:15:00Z'
          },
          {
            id: '3',
            name: 'FTP Server',
            type: 'ftp',
            status: 'inactive',
            host: '127.0.0.1',
            port: 2121,
            interactions: 12,
            lastActivity: '1 hour ago',
            created_at: '2024-01-15T08:45:00Z'
          },
          {
            id: '4',
            name: 'Telnet Service',
            type: 'telnet',
            status: 'error',
            host: '127.0.0.1',
            port: 2323,
            interactions: 8,
            lastActivity: '3 hours ago',
            created_at: '2024-01-15T07:20:00Z'
          }
        ])
      } catch (error) {
        console.error('Failed to fetch decoys:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchDecoys()
  }, [])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-success-500'
      case 'inactive':
        return 'bg-gray-500'
      case 'error':
        return 'bg-danger-500'
      case 'pending':
        return 'bg-warning-500'
      default:
        return 'bg-gray-500'
    }
  }

  const getTypeIcon = (type: string) => {
    // Return appropriate icons for different decoy types
    switch (type.toLowerCase()) {
      case 'ssh':
        return '🔐'
      case 'http':
        return '🌐'
      case 'ftp':
        return '📁'
      case 'telnet':
        return '💻'
      case 'smb':
        return '🗂️'
      case 'mysql':
        return '🗄️'
      default:
        return '🛡️'
    }
  }

  const handleStartDecoy = async (id: string) => {
    // API call to start decoy
    console.log('Starting decoy:', id)
  }

  const handleStopDecoy = async (id: string) => {
    // API call to stop decoy
    console.log('Stopping decoy:', id)
  }

  const handleDeleteDecoy = async (id: string) => {
    // API call to delete decoy
    console.log('Deleting decoy:', id)
  }

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <div className="loading-shimmer h-8 w-32 rounded"></div>
          <div className="loading-shimmer h-10 w-24 rounded"></div>
        </div>
        <div className="space-y-4">
          {[...Array(4)].map((_, i) => (
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
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold text-white">Decoys</h2>
          <p className="mt-2 text-gray-400">
            Manage your deception network honeypots and services
          </p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="btn-primary flex items-center space-x-2"
        >
          <PlusIcon className="h-5 w-5" />
          <span>Create Decoy</span>
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-4">
        <div className="card p-4">
          <div className="text-2xl font-bold text-white">{decoys.length}</div>
          <div className="text-sm text-gray-400">Total Decoys</div>
        </div>
        <div className="card p-4">
          <div className="text-2xl font-bold text-success-500">
            {decoys.filter(d => d.status === 'active').length}
          </div>
          <div className="text-sm text-gray-400">Active</div>
        </div>
        <div className="card p-4">
          <div className="text-2xl font-bold text-gray-500">
            {decoys.filter(d => d.status === 'inactive').length}
          </div>
          <div className="text-sm text-gray-400">Inactive</div>
        </div>
        <div className="card p-4">
          <div className="text-2xl font-bold text-danger-500">
            {decoys.filter(d => d.status === 'error').length}
          </div>
          <div className="text-sm text-gray-400">Errors</div>
        </div>
      </div>

      {/* Decoys List */}
      <div className="card overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-700">
          <h3 className="text-lg font-medium text-white">Deployed Decoys</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-700">
                            <thead className="bg-gray-900">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Name
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Type
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Endpoint
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Interactions
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Last Activity
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-gray-900 divide-y divide-gray-700">
              {decoys.map((decoy) => (
                                  <tr key={decoy.id} className="hover:bg-gray-700">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <span className="text-2xl mr-3">{getTypeIcon(decoy.type)}</span>
                      <div>
                        <div className="text-sm font-medium text-white">{decoy.name}</div>
                        <div className="text-sm text-gray-400">ID: {decoy.id}</div>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-500/10 text-blue-400">
                      {decoy.type.toUpperCase()}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <div className={clsx(
                        'h-2 w-2 rounded-full mr-2',
                        getStatusColor(decoy.status)
                      )}></div>
                      <span className="text-sm text-white capitalize">{decoy.status}</span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                    {decoy.host}:{decoy.port}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-white">
                    {decoy.interactions}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-400">
                    {decoy.lastActivity}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                    <div className="flex items-center justify-end space-x-2">
                      <button
                        onClick={() => console.log('View details:', decoy.id)}
                        className="text-blue-400 hover:text-blue-300"
                        title="View details"
                      >
                        <EyeIcon className="h-5 w-5" />
                      </button>
                      <button
                        onClick={() => console.log('Configure:', decoy.id)}
                        className="text-gray-400 hover:text-gray-300"
                        title="Configure"
                      >
                        <Cog6ToothIcon className="h-5 w-5" />
                      </button>
                      {decoy.status === 'active' ? (
                        <button
                          onClick={() => handleStopDecoy(decoy.id)}
                          className="text-warning-400 hover:text-warning-300"
                          title="Stop decoy"
                        >
                          <StopIcon className="h-5 w-5" />
                        </button>
                      ) : (
                        <button
                          onClick={() => handleStartDecoy(decoy.id)}
                          className="text-success-400 hover:text-success-300"
                          title="Start decoy"
                        >
                          <PlayIcon className="h-5 w-5" />
                        </button>
                      )}
                      <button
                        onClick={() => handleDeleteDecoy(decoy.id)}
                        className="text-danger-400 hover:text-danger-300"
                        title="Delete decoy"
                      >
                        <TrashIcon className="h-5 w-5" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Create Modal Placeholder */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-75 flex items-center justify-center z-50">
          <div className="card p-6 max-w-md w-full mx-4">
            <h3 className="text-lg font-medium text-white mb-4">Create New Decoy</h3>
            <p className="text-gray-400 mb-4">
              Decoy creation form will be implemented here.
            </p>
            <div className="flex justify-end space-x-3">
              <button
                onClick={() => setShowCreateModal(false)}
                className="btn-secondary"
              >
                Cancel
              </button>
              <button className="btn-primary">Create</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
} 