import { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import {
  HomeIcon,
  ShieldCheckIcon,
  EyeIcon,
  ChartBarIcon,
  CogIcon,
  Bars3Icon,
  XMarkIcon,
  ArrowRightOnRectangleIcon,
  UserIcon,
} from '@heroicons/react/24/outline'
import clsx from 'clsx'

interface LayoutProps {
  children: React.ReactNode
  user?: any
  onLogout?: () => void
}

const navigation = [
  { name: 'Dashboard', href: '/', icon: HomeIcon },
  { name: 'Decoys', href: '/decoys', icon: ShieldCheckIcon },
  { name: 'Monitoring', href: '/monitoring', icon: EyeIcon },
  { name: 'Intelligence', href: '/intelligence', icon: ChartBarIcon },
  { name: 'Settings', href: '/settings', icon: CogIcon },
]

export default function Layout({ children, user, onLogout }: LayoutProps) {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const location = useLocation()

  return (
    <div className="min-h-screen bg-gray-900">
      {/* Mobile sidebar */}
      <div className={clsx(
        'fixed inset-0 z-50 lg:hidden',
        sidebarOpen ? 'block' : 'hidden'
      )}>
        <div className="fixed inset-0 bg-gray-600 bg-opacity-75" onClick={() => setSidebarOpen(false)} />
        <div className="fixed inset-y-0 left-0 flex w-64 flex-col bg-gray-900">
          <div className="flex h-16 items-center justify-between px-4">
            <div className="flex items-center">
              <ShieldCheckIcon className="h-8 w-8 text-blue-500" />
              <span className="ml-2 text-xl font-bold text-white">AADN</span>
            </div>
            <button
              onClick={() => setSidebarOpen(false)}
              className="text-gray-400 hover:text-white"
              title="Close sidebar"
            >
              <XMarkIcon className="h-6 w-6" />
            </button>
          </div>
          <nav className="flex-1 space-y-1 px-2 py-4">
            {navigation.map((item) => {
              const isActive = location.pathname === item.href
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  className={clsx(
                    'group flex items-center px-2 py-2 text-sm font-medium rounded-md',
                    isActive
                      ? 'bg-blue-700 text-white'
                      : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                  )}
                  onClick={() => setSidebarOpen(false)}
                >
                  <item.icon className="mr-3 h-6 w-6" />
                  {item.name}
                </Link>
              )
            })}
          </nav>
        </div>
      </div>

      {/* Desktop sidebar */}
      <div className="hidden lg:fixed lg:inset-y-0 lg:flex lg:w-64 lg:flex-col">
        <div className="flex flex-col flex-grow bg-gray-900 pt-5 pb-4 overflow-y-auto">
          <div className="flex items-center flex-shrink-0 px-4">
            <ShieldCheckIcon className="h-8 w-8 text-blue-500" />
            <span className="ml-2 text-xl font-bold text-white">AADN</span>
          </div>
          <nav className="mt-8 flex-1 flex flex-col divide-y divide-gray-700 overflow-y-auto">
            <div className="px-2 space-y-1">
              {navigation.map((item) => {
                const isActive = location.pathname === item.href
                return (
                  <Link
                    key={item.name}
                    to={item.href}
                    className={clsx(
                      'group flex items-center px-2 py-2 text-sm font-medium rounded-md',
                      isActive
                        ? 'bg-blue-700 text-white'
                        : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                    )}
                  >
                    <item.icon className="mr-3 h-6 w-6" />
                    {item.name}
                  </Link>
                )
              })}
            </div>
          </nav>
        </div>
      </div>

      {/* Main content */}
      <div className="lg:pl-64 flex flex-col flex-1">
        {/* Top bar */}
        <div className="sticky top-0 z-10 flex h-16 bg-gray-900 shadow-sm border-b border-gray-700">
          <button
            type="button"
            className="px-4 border-r border-gray-700 text-gray-400 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-blue-500 lg:hidden"
            onClick={() => setSidebarOpen(true)}
            title="Open sidebar"
          >
            <Bars3Icon className="h-6 w-6" />
          </button>
          <div className="flex-1 px-4 flex justify-between items-center">
            <div className="flex-1 flex">
              <h1 className="text-2xl font-semibold text-white">
                Adaptive AI-Driven Deception Network
              </h1>
            </div>
            <div className="ml-4 flex items-center md:ml-6 space-x-4">
              <div className="flex items-center space-x-2">
                <div className="h-3 w-3 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-sm text-gray-300">System Online</span>
              </div>
              
              {user && (
                <div className="flex items-center space-x-3">
                  <div className="flex items-center space-x-2">
                    <UserIcon className="h-5 w-5 text-gray-400" />
                    <span className="text-sm text-gray-300">{user.username}</span>
                    <span className="text-xs text-gray-500 bg-gray-700 px-2 py-1 rounded">
                      {user.role}
                    </span>
                  </div>
                  
                  {onLogout && (
                    <button
                      onClick={onLogout}
                      className="flex items-center space-x-1 text-gray-400 hover:text-white transition-colors"
                      title="Logout"
                    >
                      <ArrowRightOnRectangleIcon className="h-5 w-5" />
                      <span className="text-sm">Logout</span>
                    </button>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Page content */}
        <main className="flex-1">
          <div className="py-6">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 md:px-8">
              {children}
            </div>
          </div>
        </main>
      </div>
    </div>
  )
} 