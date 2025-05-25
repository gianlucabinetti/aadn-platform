import { ChartBarIcon, ExclamationTriangleIcon, InformationCircleIcon } from '@heroicons/react/24/outline'

export default function Intelligence() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold text-white">Threat Intelligence</h2>
        <p className="mt-2 text-gray-400">
          AI-powered threat analysis and intelligence gathering
        </p>
      </div>

      {/* Coming Soon Notice */}
      <div className="card p-8 text-center">
        <div className="flex justify-center mb-4">
          <ChartBarIcon className="h-16 w-16 text-blue-500" />
        </div>
        <h3 className="text-xl font-semibold text-white mb-2">
          Advanced Threat Intelligence
        </h3>
        <p className="text-gray-400 mb-6 max-w-2xl mx-auto">
          This section will feature AI-powered threat analysis, TTP classification, 
          attacker behavior modeling, and predictive intelligence capabilities. 
          These features are planned for Phase 2 of the AADN development.
        </p>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
          <div className="card p-6">
            <div className="flex justify-center mb-3">
              <ExclamationTriangleIcon className="h-8 w-8 text-warning-500" />
            </div>
            <h4 className="text-lg font-medium text-white mb-2">TTP Analysis</h4>
            <p className="text-sm text-gray-400">
              Automated classification of attacker tactics, techniques, and procedures
            </p>
          </div>
          
          <div className="card p-6">
            <div className="flex justify-center mb-3">
              <ChartBarIcon className="h-8 w-8 text-blue-500" />
            </div>
            <h4 className="text-lg font-medium text-white mb-2">Behavior Modeling</h4>
            <p className="text-sm text-gray-400">
              Machine learning models to understand and predict attacker behavior
            </p>
          </div>
          
          <div className="card p-6">
            <div className="flex justify-center mb-3">
              <InformationCircleIcon className="h-8 w-8 text-success-500" />
            </div>
            <h4 className="text-lg font-medium text-white mb-2">Threat Feeds</h4>
            <p className="text-sm text-gray-400">
              Integration with external threat intelligence feeds and IOC databases
            </p>
          </div>
        </div>
      </div>
    </div>
  )
} 