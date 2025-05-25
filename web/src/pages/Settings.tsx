import { useState } from 'react'
import { CogIcon, BellIcon, ShieldCheckIcon, CircleStackIcon } from '@heroicons/react/24/outline'

export default function Settings() {
  const [notifications, setNotifications] = useState(true)
  const [autoResponse, setAutoResponse] = useState(false)
  const [logLevel, setLogLevel] = useState('info')

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold text-white">Settings</h2>
        <p className="mt-2 text-gray-400">
          Configure your AADN system preferences and security settings
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* General Settings */}
        <div className="card p-6">
          <div className="flex items-center mb-4">
                            <CogIcon className="h-6 w-6 text-blue-500 mr-2" />
            <h3 className="text-lg font-medium text-white">General Settings</h3>
          </div>
          
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <label className="text-sm font-medium text-white">Enable Notifications</label>
                <p className="text-xs text-gray-400">Receive alerts for new interactions</p>
              </div>
              <button
                onClick={() => setNotifications(!notifications)}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  notifications ? 'bg-blue-600' : 'bg-gray-600'
                }`}
                aria-label="Toggle notifications"
                title="Toggle notifications"
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    notifications ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>

            <div className="flex items-center justify-between">
              <div>
                <label className="text-sm font-medium text-white">Auto Response</label>
                <p className="text-xs text-gray-400">Automatically respond to threats</p>
              </div>
              <button
                onClick={() => setAutoResponse(!autoResponse)}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  autoResponse ? 'bg-blue-600' : 'bg-gray-600'
                }`}
                aria-label="Toggle auto response"
                title="Toggle auto response"
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    autoResponse ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>

            <div>
              <label className="block text-sm font-medium text-white mb-2">Log Level</label>
              <select
                value={logLevel}
                onChange={(e) => setLogLevel(e.target.value)}
                className="input-field w-full"
                aria-label="Log Level"
                title="Select log level"
              >
                <option value="debug">Debug</option>
                <option value="info">Info</option>
                <option value="warning">Warning</option>
                <option value="error">Error</option>
              </select>
            </div>
          </div>
        </div>

        {/* Security Settings */}
        <div className="card p-6">
          <div className="flex items-center mb-4">
            <ShieldCheckIcon className="h-6 w-6 text-success-500 mr-2" />
            <h3 className="text-lg font-medium text-white">Security Settings</h3>
          </div>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-white mb-2">Session Timeout</label>
              <select className="input-field w-full" aria-label="Session Timeout" title="Select session timeout duration">
                <option value="30">30 minutes</option>
                <option value="60">1 hour</option>
                <option value="240">4 hours</option>
                <option value="480">8 hours</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-white mb-2">API Rate Limit</label>
              <input
                type="number"
                defaultValue={100}
                className="input-field w-full"
                placeholder="Requests per minute"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-white mb-2">Allowed IP Ranges</label>
              <textarea
                className="input-field w-full h-20"
                placeholder="192.168.1.0/24&#10;10.0.0.0/8"
                defaultValue="192.168.1.0/24&#10;10.0.0.0/8"
              />
            </div>
          </div>
        </div>

        {/* Alert Settings */}
        <div className="card p-6">
          <div className="flex items-center mb-4">
            <BellIcon className="h-6 w-6 text-warning-500 mr-2" />
            <h3 className="text-lg font-medium text-white">Alert Settings</h3>
          </div>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-white mb-2">Email Notifications</label>
              <input
                type="email"
                className="input-field w-full"
                placeholder="admin@company.com"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-white mb-2">Webhook URL</label>
              <input
                type="url"
                className="input-field w-full"
                placeholder="https://hooks.slack.com/..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-white mb-2">Alert Threshold</label>
              <select className="input-field w-full" aria-label="Alert Threshold" title="Select alert threshold level">
                <option value="low">Low - All interactions</option>
                <option value="medium">Medium - Suspicious activity</option>
                <option value="high">High - Critical threats only</option>
              </select>
            </div>
          </div>
        </div>

        {/* Database Settings */}
        <div className="card p-6">
          <div className="flex items-center mb-4">
                            <CircleStackIcon className="h-6 w-6 text-blue-500 mr-2" />
            <h3 className="text-lg font-medium text-white">Database Settings</h3>
          </div>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-white mb-2">Data Retention</label>
              <select className="input-field w-full" aria-label="Data Retention" title="Select data retention period">
                <option value="30">30 days</option>
                <option value="90">90 days</option>
                <option value="180">6 months</option>
                <option value="365">1 year</option>
                <option value="0">Unlimited</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-white mb-2">Backup Frequency</label>
              <select className="input-field w-full" aria-label="Backup Frequency" title="Select backup frequency">
                <option value="daily">Daily</option>
                <option value="weekly">Weekly</option>
                <option value="monthly">Monthly</option>
              </select>
            </div>

            <div className="pt-4">
              <button className="btn-primary mr-3">Backup Now</button>
              <button className="btn-secondary">Test Connection</button>
            </div>
          </div>
        </div>
      </div>

      {/* Save Button */}
      <div className="flex justify-end">
        <button className="btn-primary">Save Settings</button>
      </div>
    </div>
  )
} 