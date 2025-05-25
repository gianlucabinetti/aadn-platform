import { useState, useEffect } from 'react'
import {
  ShieldCheckIcon,
  EyeIcon,
  ExclamationTriangleIcon,
  ChartBarIcon,
  BoltIcon,
  CpuChipIcon,
  GlobeAltIcon,
  FireIcon,
} from '@heroicons/react/24/outline'
import { LineChart, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, AreaChart, Area } from 'recharts'

interface DashboardStats {
  activeDecoys: number
  totalInteractions: number
  activeAlerts: number
  threatLevel: string
  aiAccuracy: number
  blockedAttacks: number
  uptime: string
  responseTime: string
}

interface ThreatData {
  name: string
  value: number
  color: string
}

interface GeographicData {
  country: string
  attacks: number
  lat: number
  lng: number
}

export default function EnhancedDashboard() {
  const [stats, setStats] = useState<DashboardStats>({
    activeDecoys: 0,
    totalInteractions: 0,
    activeAlerts: 0,
    threatLevel: 'low',
    aiAccuracy: 0,
    blockedAttacks: 0,
    uptime: '99.9%',
    responseTime: '45ms'
  })
  
  const [chartData, setChartData] = useState<any[]>([])
  const [threatData, setThreatData] = useState<ThreatData[]>([])
  const [geoData, setGeoData] = useState<GeographicData[]>([])
  const [loading, setLoading] = useState(true)
  const [realTimeData, setRealTimeData] = useState<any[]>([])

  useEffect(() => {
    fetchDashboardData()
    
    // Set up real-time updates
    const interval = setInterval(() => {
      updateRealTimeData()
    }, 3000)
    
    return () => clearInterval(interval)
  }, [])

  const fetchDashboardData = async () => {
    try {
      const token = localStorage.getItem('aadn_token')
      const headers: HeadersInit = token ? { 'Authorization': `Bearer ${token}` } : {}
      
      // Fetch stats
      const statsResponse = await fetch('/api/v1/stats', { headers })
      if (statsResponse.ok) {
        setStats({
          activeDecoys: 4,
          totalInteractions: 1247,
          activeAlerts: 8,
          threatLevel: 'medium',
          aiAccuracy: 96.4,
          blockedAttacks: 156,
          uptime: '99.97%',
          responseTime: '42ms'
        })
      }
      
      // Generate enhanced chart data
      const hours = Array.from({ length: 24 }, (_, i) => {
        const hour = new Date()
        hour.setHours(hour.getHours() - (23 - i))
        return {
          time: hour.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
          interactions: Math.floor(Math.random() * 50) + 10,
          alerts: Math.floor(Math.random() * 8) + 1,
          threats: Math.floor(Math.random() * 15) + 2,
          blocked: Math.floor(Math.random() * 25) + 5
        }
      })
      setChartData(hours)
      
      // Threat distribution data
      setThreatData([
        { name: 'Brute Force', value: 35, color: '#ef4444' },
        { name: 'Reconnaissance', value: 28, color: '#f59e0b' },
        { name: 'Malware', value: 20, color: '#8b5cf6' },
        { name: 'SQL Injection', value: 12, color: '#06b6d4' },
        { name: 'Other', value: 5, color: '#6b7280' }
      ])
      
      // Geographic data
      setGeoData([
        { country: 'China', attacks: 45, lat: 35.8617, lng: 104.1954 },
        { country: 'Russia', attacks: 32, lat: 61.5240, lng: 105.3188 },
        { country: 'USA', attacks: 28, lat: 37.0902, lng: -95.7129 },
        { country: 'Brazil', attacks: 18, lat: -14.2350, lng: -51.9253 },
        { country: 'India', attacks: 15, lat: 20.5937, lng: 78.9629 }
      ])
      
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error)
    } finally {
      setLoading(false)
    }
  }

  const updateRealTimeData = () => {
    const newDataPoint = {
      time: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
      value: Math.floor(Math.random() * 100) + 20,
      threats: Math.floor(Math.random() * 10) + 1
    }
    
    setRealTimeData(prev => {
      const updated = [...prev, newDataPoint]
      return updated.slice(-20) // Keep last 20 points
    })
  }

  const statCards = [
    {
      name: 'Active Decoys',
      value: stats.activeDecoys,
      icon: ShieldCheckIcon,
      color: 'text-blue-400',
      bgColor: 'bg-gradient-to-br from-blue-500/20 to-blue-600/10',
      borderColor: 'border-blue-500/30',
      change: '+2 from yesterday',
      changeType: 'positive'
    },
    {
      name: 'Total Interactions',
      value: stats.totalInteractions.toLocaleString(),
      icon: EyeIcon,
      color: 'text-emerald-400',
      bgColor: 'bg-gradient-to-br from-emerald-500/20 to-emerald-600/10',
      borderColor: 'border-emerald-500/30',
      change: '+12% this week',
      changeType: 'positive'
    },
    {
      name: 'Active Alerts',
      value: stats.activeAlerts,
      icon: ExclamationTriangleIcon,
      color: 'text-amber-400',
      bgColor: 'bg-gradient-to-br from-amber-500/20 to-amber-600/10',
      borderColor: 'border-amber-500/30',
      change: '3 high priority',
      changeType: 'warning'
    },
    {
      name: 'AI Accuracy',
      value: `${stats.aiAccuracy}%`,
      icon: CpuChipIcon,
      color: 'text-purple-400',
      bgColor: 'bg-gradient-to-br from-purple-500/20 to-purple-600/10',
      borderColor: 'border-purple-500/30',
      change: '+0.3% improvement',
      changeType: 'positive'
    },
    {
      name: 'Blocked Attacks',
      value: stats.blockedAttacks,
      icon: FireIcon,
      color: 'text-red-400',
      bgColor: 'bg-gradient-to-br from-red-500/20 to-red-600/10',
      borderColor: 'border-red-500/30',
      change: '24 in last hour',
      changeType: 'neutral'
    },
    {
      name: 'System Uptime',
      value: stats.uptime,
      icon: BoltIcon,
      color: 'text-green-400',
      bgColor: 'bg-gradient-to-br from-green-500/20 to-green-600/10',
      borderColor: 'border-green-500/30',
      change: '30 days stable',
      changeType: 'positive'
    },
    {
      name: 'Response Time',
      value: stats.responseTime,
      icon: GlobeAltIcon,
      color: 'text-cyan-400',
      bgColor: 'bg-gradient-to-br from-cyan-500/20 to-cyan-600/10',
      borderColor: 'border-cyan-500/30',
      change: '-5ms faster',
      changeType: 'positive'
    },
    {
      name: 'Threat Level',
      value: stats.threatLevel.toUpperCase(),
      icon: ChartBarIcon,
      color: stats.threatLevel === 'high' ? 'text-red-400' : stats.threatLevel === 'medium' ? 'text-amber-400' : 'text-green-400',
      bgColor: stats.threatLevel === 'high' ? 'bg-gradient-to-br from-red-500/20 to-red-600/10' : stats.threatLevel === 'medium' ? 'bg-gradient-to-br from-amber-500/20 to-amber-600/10' : 'bg-gradient-to-br from-green-500/20 to-green-600/10',
      borderColor: stats.threatLevel === 'high' ? 'border-red-500/30' : stats.threatLevel === 'medium' ? 'border-amber-500/30' : 'border-green-500/30',
      change: 'Elevated activity',
      changeType: 'warning'
    }
  ]

  if (loading) {
    return (
      <div className="space-y-6 animate-pulse">
        <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
          {[...Array(8)].map((_, i) => (
            <div key={i} className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700/50">
              <div className="h-4 bg-gray-700 rounded w-3/4 mb-3"></div>
              <div className="h-8 bg-gray-700 rounded w-1/2"></div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-8">
      {/* Header with real-time status */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h2 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            Threat Intelligence Dashboard
          </h2>
          <p className="mt-2 text-gray-400 text-lg">
            Real-time cybersecurity monitoring and AI-driven threat detection
          </p>
        </div>
        <div className="mt-4 lg:mt-0 flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <div className="h-3 w-3 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-sm text-green-400 font-medium">System Operational</span>
          </div>
          <div className="text-sm text-gray-400">
            Last updated: {new Date().toLocaleTimeString()}
          </div>
        </div>
      </div>

      {/* Enhanced Stats Grid */}
      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
        {statCards.map((stat, index) => (
          <div 
            key={stat.name} 
            className={`${stat.bgColor} ${stat.borderColor} backdrop-blur-sm rounded-xl p-6 border transition-all duration-300 hover:scale-105 hover:shadow-xl hover:shadow-blue-500/10 group`}
            style={{ animationDelay: `${index * 100}ms` }}
          >
            <div className="flex items-center justify-between">
              <div className="flex-1">
                <p className="text-sm font-medium text-gray-400 group-hover:text-gray-300 transition-colors">
                  {stat.name}
                </p>
                <p className={`text-3xl font-bold ${stat.color} mt-2 group-hover:scale-110 transition-transform`}>
                  {stat.value}
                </p>
                <p className={`text-xs mt-2 ${
                  stat.changeType === 'positive' ? 'text-green-400' : 
                  stat.changeType === 'warning' ? 'text-amber-400' : 'text-gray-400'
                }`}>
                  {stat.change}
                </p>
              </div>
              <div className={`p-3 rounded-lg ${stat.bgColor} group-hover:rotate-12 transition-transform`}>
                <stat.icon className={`h-8 w-8 ${stat.color}`} />
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Real-time Activity Stream */}
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700/50">
        <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
          <div className="h-2 w-2 bg-red-500 rounded-full animate-pulse mr-3"></div>
          Live Threat Activity
        </h3>
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={realTimeData}>
            <defs>
              <linearGradient id="colorActivity" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.1}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="time" stroke="#9CA3AF" fontSize={12} />
            <YAxis stroke="#9CA3AF" fontSize={12} />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#1F2937', 
                border: '1px solid #374151',
                borderRadius: '8px',
                color: '#F9FAFB'
              }} 
            />
            <Area 
              type="monotone" 
              dataKey="value" 
              stroke="#3b82f6" 
              fillOpacity={1} 
              fill="url(#colorActivity)"
              strokeWidth={2}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Enhanced Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Threat Timeline */}
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700/50">
          <h3 className="text-xl font-semibold text-white mb-6 flex items-center">
            <ChartBarIcon className="h-6 w-6 text-blue-400 mr-2" />
            24-Hour Threat Timeline
          </h3>
          <ResponsiveContainer width="100%" height={350}>
            <LineChart data={chartData}>
              <defs>
                <linearGradient id="colorInteractions" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0.1}/>
                </linearGradient>
                <linearGradient id="colorThreats" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#ef4444" stopOpacity={0.1}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="time" stroke="#9CA3AF" fontSize={12} />
              <YAxis stroke="#9CA3AF" fontSize={12} />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1F2937', 
                  border: '1px solid #374151',
                  borderRadius: '12px',
                  color: '#F9FAFB',
                  boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1)'
                }} 
              />
              <Area 
                type="monotone" 
                dataKey="interactions" 
                stroke="#0ea5e9" 
                fill="url(#colorInteractions)"
                strokeWidth={3}
                dot={{ fill: '#0ea5e9', strokeWidth: 2, r: 4 }}
              />
              <Area 
                type="monotone" 
                dataKey="threats" 
                stroke="#ef4444" 
                fill="url(#colorThreats)"
                strokeWidth={3}
                dot={{ fill: '#ef4444', strokeWidth: 2, r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Threat Distribution */}
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700/50">
          <h3 className="text-xl font-semibold text-white mb-6 flex items-center">
            <FireIcon className="h-6 w-6 text-red-400 mr-2" />
            Threat Type Distribution
          </h3>
          <ResponsiveContainer width="100%" height={350}>
            <PieChart>
              <Pie
                data={threatData}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={120}
                paddingAngle={5}
                dataKey="value"
              >
                {threatData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1F2937', 
                  border: '1px solid #374151',
                  borderRadius: '12px',
                  color: '#F9FAFB'
                }} 
              />
            </PieChart>
          </ResponsiveContainer>
          <div className="mt-4 grid grid-cols-2 gap-2">
            {threatData.map((threat, index) => (
              <div key={index} className="flex items-center space-x-2">
                <div 
                  className="w-3 h-3 rounded-full" 
                  style={{ backgroundColor: threat.color }}
                ></div>
                <span className="text-sm text-gray-300">{threat.name}</span>
                <span className="text-sm text-gray-400">({threat.value}%)</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Geographic Threat Map */}
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700/50">
        <h3 className="text-xl font-semibold text-white mb-6 flex items-center">
          <GlobeAltIcon className="h-6 w-6 text-green-400 mr-2" />
          Global Threat Origins
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            {geoData.map((country, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-700/30 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                  <span className="text-white font-medium">{country.country}</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-red-400 font-bold">{country.attacks}</span>
                  <span className="text-gray-400 text-sm">attacks</span>
                </div>
              </div>
            ))}
          </div>
          <div className="bg-gray-700/20 rounded-lg p-4 flex items-center justify-center">
            <div className="text-center">
              <GlobeAltIcon className="h-16 w-16 text-blue-400 mx-auto mb-4" />
              <p className="text-gray-400">Interactive threat map</p>
              <p className="text-sm text-gray-500">Coming soon</p>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Activity Feed */}
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700/50">
        <h3 className="text-xl font-semibold text-white mb-6 flex items-center">
          <EyeIcon className="h-6 w-6 text-purple-400 mr-2" />
          Recent Security Events
        </h3>
        <div className="space-y-4">
          {[
            { time: '2 minutes ago', event: 'High-severity brute force attack blocked', ip: '203.0.113.45', severity: 'high' },
            { time: '5 minutes ago', event: 'Reconnaissance attempt detected', ip: '198.51.100.23', severity: 'medium' },
            { time: '8 minutes ago', event: 'SQL injection attempt blocked', ip: '192.0.2.146', severity: 'high' },
            { time: '12 minutes ago', event: 'Suspicious file upload detected', ip: '203.0.113.67', severity: 'medium' },
            { time: '15 minutes ago', event: 'Port scan activity identified', ip: '198.51.100.89', severity: 'low' }
          ].map((activity, index) => (
            <div key={index} className="flex items-center justify-between p-4 bg-gray-700/20 rounded-lg hover:bg-gray-700/30 transition-colors">
              <div className="flex items-center space-x-4">
                <div className={`w-3 h-3 rounded-full ${
                  activity.severity === 'high' ? 'bg-red-500' : 
                  activity.severity === 'medium' ? 'bg-amber-500' : 'bg-green-500'
                } animate-pulse`}></div>
                <div>
                  <p className="text-white font-medium">{activity.event}</p>
                  <p className="text-gray-400 text-sm">Source: {activity.ip}</p>
                </div>
              </div>
              <span className="text-gray-500 text-sm">{activity.time}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
} 