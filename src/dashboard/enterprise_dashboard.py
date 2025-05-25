"""
AADN Enterprise Dashboard Module
Advanced real-time security dashboard with executive reporting and analytics
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
import logging
from dataclasses import dataclass, asdict
import base64
import hashlib
import secrets

logger = logging.getLogger(__name__)

@dataclass
class DashboardMetric:
    """Dashboard metric data structure"""
    metric_id: str
    name: str
    value: float
    unit: str
    trend: str  # up, down, stable
    change_percentage: float
    timestamp: datetime
    category: str
    severity: str
    threshold_status: str  # normal, warning, critical

@dataclass
class SecurityAlert:
    """Security alert for dashboard"""
    alert_id: str
    title: str
    description: str
    severity: str
    category: str
    timestamp: datetime
    source: str
    affected_assets: List[str]
    status: str  # new, investigating, resolved
    assigned_to: Optional[str]

@dataclass
class ThreatIndicator:
    """Threat indicator for dashboard"""
    indicator_id: str
    type: str
    value: str
    threat_level: str
    confidence: float
    first_seen: datetime
    last_seen: datetime
    source: str
    tags: List[str]

class RealTimeAnalytics:
    """Real-time analytics engine for dashboard"""
    
    def __init__(self):
        self.metrics_buffer = deque(maxlen=1000)
        self.alert_buffer = deque(maxlen=500)
        self.threat_buffer = deque(maxlen=300)
        self.performance_metrics = {}
        self.security_metrics = {}
        self.business_metrics = {}
        
        # Time series data
        self.time_series_data = {
            'threat_volume': deque(maxlen=288),  # 24 hours at 5-minute intervals
            'attack_attempts': deque(maxlen=288),
            'system_performance': deque(maxlen=288),
            'user_activity': deque(maxlen=288),
            'network_traffic': deque(maxlen=288)
        }
        
        # Initialize with sample data
        self._initialize_sample_data()
    
    def _initialize_sample_data(self):
        """Initialize with sample data for demonstration"""
        current_time = datetime.utcnow()
        
        # Generate sample time series data
        for i in range(288):  # 24 hours of data
            timestamp = current_time - timedelta(minutes=i*5)
            
            # Simulate realistic patterns
            hour = timestamp.hour
            base_threat = 10 + 5 * np.sin(hour * np.pi / 12)  # Daily pattern
            noise = np.random.normal(0, 2)
            
            self.time_series_data['threat_volume'].appendleft({
                'timestamp': timestamp.isoformat(),
                'value': max(0, base_threat + noise)
            })
            
            self.time_series_data['attack_attempts'].appendleft({
                'timestamp': timestamp.isoformat(),
                'value': max(0, np.random.poisson(3) + (5 if 9 <= hour <= 17 else 0))
            })
            
            self.time_series_data['system_performance'].appendleft({
                'timestamp': timestamp.isoformat(),
                'value': 85 + 10 * np.sin(hour * np.pi / 12) + np.random.normal(0, 3)
            })
    
    def add_metric(self, metric: DashboardMetric):
        """Add a new metric to the analytics engine"""
        self.metrics_buffer.append(metric)
        
        # Update category metrics
        if metric.category == 'security':
            self.security_metrics[metric.metric_id] = metric
        elif metric.category == 'performance':
            self.performance_metrics[metric.metric_id] = metric
        elif metric.category == 'business':
            self.business_metrics[metric.metric_id] = metric
    
    def add_alert(self, alert: SecurityAlert):
        """Add a new security alert"""
        self.alert_buffer.append(alert)
    
    def add_threat_indicator(self, indicator: ThreatIndicator):
        """Add a new threat indicator"""
        self.threat_buffer.append(indicator)
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get current real-time metrics"""
        current_time = datetime.utcnow()
        
        # Calculate current metrics
        active_threats = len([t for t in self.threat_buffer 
                            if current_time - t.last_seen < timedelta(hours=1)])
        
        critical_alerts = len([a for a in self.alert_buffer 
                             if a.severity == 'critical' and a.status != 'resolved'])
        
        system_health = np.mean([m.value for m in self.performance_metrics.values()]) if self.performance_metrics else 95.0
        
        # Calculate trends
        threat_trend = self._calculate_trend('threat_volume')
        performance_trend = self._calculate_trend('system_performance')
        
        return {
            'timestamp': current_time.isoformat(),
            'active_threats': active_threats,
            'critical_alerts': critical_alerts,
            'system_health': round(system_health, 2),
            'threat_trend': threat_trend,
            'performance_trend': performance_trend,
            'total_events_24h': len(self.metrics_buffer),
            'security_score': self._calculate_security_score(),
            'uptime_percentage': 99.97,
            'response_time_avg': 0.23
        }
    
    def _calculate_trend(self, metric_name: str) -> str:
        """Calculate trend for a metric"""
        if metric_name not in self.time_series_data:
            return 'stable'
        
        data = list(self.time_series_data[metric_name])
        if len(data) < 10:
            return 'stable'
        
        recent_avg = np.mean([d['value'] for d in data[:10]])
        older_avg = np.mean([d['value'] for d in data[10:20]])
        
        if recent_avg > older_avg * 1.1:
            return 'up'
        elif recent_avg < older_avg * 0.9:
            return 'down'
        else:
            return 'stable'
    
    def _calculate_security_score(self) -> float:
        """Calculate overall security score"""
        base_score = 100.0
        
        # Deduct points for active threats
        active_threats = len([t for t in self.threat_buffer 
                            if datetime.utcnow() - t.last_seen < timedelta(hours=1)])
        base_score -= active_threats * 2
        
        # Deduct points for critical alerts
        critical_alerts = len([a for a in self.alert_buffer 
                             if a.severity == 'critical' and a.status != 'resolved'])
        base_score -= critical_alerts * 5
        
        # Deduct points for high-confidence threats
        high_confidence_threats = len([t for t in self.threat_buffer 
                                     if t.confidence > 0.8])
        base_score -= high_confidence_threats * 3
        
        return max(0.0, min(100.0, base_score))

class ExecutiveReporting:
    """Executive-level reporting and analytics"""
    
    def __init__(self, analytics_engine: RealTimeAnalytics):
        self.analytics = analytics_engine
        self.report_cache = {}
        self.kpi_definitions = self._define_kpis()
    
    def _define_kpis(self) -> Dict[str, Dict]:
        """Define key performance indicators"""
        return {
            'security_posture': {
                'name': 'Security Posture Score',
                'description': 'Overall security effectiveness',
                'target': 95.0,
                'unit': '%',
                'category': 'security'
            },
            'threat_detection_rate': {
                'name': 'Threat Detection Rate',
                'description': 'Percentage of threats detected',
                'target': 98.0,
                'unit': '%',
                'category': 'security'
            },
            'incident_response_time': {
                'name': 'Mean Time to Response',
                'description': 'Average time to respond to incidents',
                'target': 15.0,
                'unit': 'minutes',
                'category': 'operational'
            },
            'false_positive_rate': {
                'name': 'False Positive Rate',
                'description': 'Percentage of false positive alerts',
                'target': 5.0,
                'unit': '%',
                'category': 'operational'
            },
            'system_availability': {
                'name': 'System Availability',
                'description': 'Uptime percentage',
                'target': 99.9,
                'unit': '%',
                'category': 'performance'
            },
            'compliance_score': {
                'name': 'Compliance Score',
                'description': 'Regulatory compliance percentage',
                'target': 100.0,
                'unit': '%',
                'category': 'compliance'
            }
        }
    
    def generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary report"""
        current_time = datetime.utcnow()
        
        # Calculate KPIs
        kpi_values = {}
        for kpi_id, kpi_def in self.kpi_definitions.items():
            kpi_values[kpi_id] = self._calculate_kpi_value(kpi_id)
        
        # Risk assessment
        risk_assessment = self._assess_current_risks()
        
        # Trend analysis
        trend_analysis = self._analyze_trends()
        
        # Recommendations
        recommendations = self._generate_recommendations(kpi_values, risk_assessment)
        
        summary = {
            'report_timestamp': current_time.isoformat(),
            'report_period': '24_hours',
            'executive_summary': {
                'overall_security_status': self._determine_security_status(kpi_values),
                'key_achievements': self._identify_achievements(kpi_values),
                'critical_issues': self._identify_critical_issues(kpi_values, risk_assessment),
                'business_impact': self._assess_business_impact()
            },
            'kpi_dashboard': kpi_values,
            'risk_assessment': risk_assessment,
            'trend_analysis': trend_analysis,
            'recommendations': recommendations,
            'compliance_status': self._get_compliance_status(),
            'resource_utilization': self._get_resource_utilization(),
            'cost_analysis': self._get_cost_analysis()
        }
        
        return summary
    
    def _calculate_kpi_value(self, kpi_id: str) -> Dict[str, Any]:
        """Calculate specific KPI value"""
        kpi_def = self.kpi_definitions[kpi_id]
        
        if kpi_id == 'security_posture':
            current_value = self.analytics._calculate_security_score()
        elif kpi_id == 'threat_detection_rate':
            current_value = 97.8  # Simulated high detection rate
        elif kpi_id == 'incident_response_time':
            current_value = 12.5  # Simulated response time
        elif kpi_id == 'false_positive_rate':
            current_value = 3.2   # Simulated low false positive rate
        elif kpi_id == 'system_availability':
            current_value = 99.97 # Simulated high availability
        elif kpi_id == 'compliance_score':
            current_value = 98.5  # Simulated compliance score
        else:
            current_value = 85.0  # Default value
        
        target = kpi_def['target']
        variance = ((current_value - target) / target) * 100
        
        # Determine status
        if variance >= 0:
            status = 'exceeding' if variance > 5 else 'meeting'
        else:
            status = 'below' if variance < -10 else 'approaching'
        
        return {
            'name': kpi_def['name'],
            'current_value': round(current_value, 2),
            'target_value': target,
            'variance_percentage': round(variance, 2),
            'status': status,
            'unit': kpi_def['unit'],
            'category': kpi_def['category'],
            'trend': self._get_kpi_trend(kpi_id),
            'last_updated': datetime.utcnow().isoformat()
        }
    
    def _get_kpi_trend(self, kpi_id: str) -> str:
        """Get trend for specific KPI"""
        # Simulate trend calculation
        trends = ['improving', 'stable', 'declining']
        return np.random.choice(trends, p=[0.6, 0.3, 0.1])  # Bias toward improvement
    
    def _assess_current_risks(self) -> Dict[str, Any]:
        """Assess current risk levels"""
        risks = {
            'overall_risk_level': 'medium',
            'risk_categories': {
                'cyber_threats': {
                    'level': 'medium',
                    'score': 6.2,
                    'factors': ['Increased phishing attempts', 'New malware variants'],
                    'mitigation_status': 'active'
                },
                'operational_risks': {
                    'level': 'low',
                    'score': 3.1,
                    'factors': ['System dependencies', 'Staff availability'],
                    'mitigation_status': 'monitored'
                },
                'compliance_risks': {
                    'level': 'low',
                    'score': 2.8,
                    'factors': ['Regulatory changes', 'Audit findings'],
                    'mitigation_status': 'controlled'
                },
                'business_continuity': {
                    'level': 'low',
                    'score': 2.5,
                    'factors': ['Backup systems', 'Recovery procedures'],
                    'mitigation_status': 'optimized'
                }
            },
            'risk_trend': 'stable',
            'top_risks': [
                {
                    'risk_id': 'RISK-001',
                    'description': 'Advanced persistent threat campaign',
                    'probability': 0.3,
                    'impact': 'high',
                    'risk_score': 7.5,
                    'mitigation_actions': ['Enhanced monitoring', 'Threat hunting']
                },
                {
                    'risk_id': 'RISK-002',
                    'description': 'Supply chain compromise',
                    'probability': 0.2,
                    'impact': 'critical',
                    'risk_score': 8.0,
                    'mitigation_actions': ['Vendor assessment', 'Code signing verification']
                }
            ]
        }
        
        return risks
    
    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze security and operational trends"""
        return {
            'security_trends': {
                'threat_volume': {
                    'direction': 'increasing',
                    'rate': '+12%',
                    'period': '7_days',
                    'significance': 'moderate'
                },
                'attack_sophistication': {
                    'direction': 'increasing',
                    'rate': '+8%',
                    'period': '30_days',
                    'significance': 'high'
                },
                'detection_accuracy': {
                    'direction': 'improving',
                    'rate': '+3%',
                    'period': '30_days',
                    'significance': 'positive'
                }
            },
            'operational_trends': {
                'response_time': {
                    'direction': 'improving',
                    'rate': '-15%',
                    'period': '30_days',
                    'significance': 'positive'
                },
                'system_performance': {
                    'direction': 'stable',
                    'rate': '+1%',
                    'period': '7_days',
                    'significance': 'neutral'
                }
            },
            'business_trends': {
                'cost_efficiency': {
                    'direction': 'improving',
                    'rate': '+5%',
                    'period': '90_days',
                    'significance': 'positive'
                },
                'user_satisfaction': {
                    'direction': 'stable',
                    'rate': '+2%',
                    'period': '30_days',
                    'significance': 'neutral'
                }
            }
        }
    
    def _generate_recommendations(self, kpi_values: Dict, risk_assessment: Dict) -> List[Dict]:
        """Generate strategic recommendations"""
        recommendations = []
        
        # Analyze KPIs for recommendations
        for kpi_id, kpi_data in kpi_values.items():
            if kpi_data['status'] == 'below':
                recommendations.append({
                    'priority': 'high',
                    'category': 'performance_improvement',
                    'title': f"Improve {kpi_data['name']}",
                    'description': f"Current performance is {kpi_data['variance_percentage']:.1f}% below target",
                    'actions': [
                        'Conduct root cause analysis',
                        'Implement targeted improvements',
                        'Increase monitoring frequency'
                    ],
                    'timeline': '30_days',
                    'expected_impact': 'medium'
                })
        
        # Risk-based recommendations
        for risk in risk_assessment['top_risks']:
            if risk['risk_score'] > 7.0:
                recommendations.append({
                    'priority': 'critical',
                    'category': 'risk_mitigation',
                    'title': f"Mitigate {risk['description']}",
                    'description': f"High-impact risk with score {risk['risk_score']}",
                    'actions': risk['mitigation_actions'],
                    'timeline': '14_days',
                    'expected_impact': 'high'
                })
        
        # Strategic recommendations
        recommendations.extend([
            {
                'priority': 'medium',
                'category': 'strategic_enhancement',
                'title': 'Implement AI-powered threat prediction',
                'description': 'Enhance proactive threat detection capabilities',
                'actions': [
                    'Deploy machine learning models',
                    'Integrate threat intelligence feeds',
                    'Train security team on new tools'
                ],
                'timeline': '90_days',
                'expected_impact': 'high'
            },
            {
                'priority': 'low',
                'category': 'operational_efficiency',
                'title': 'Automate routine security tasks',
                'description': 'Reduce manual workload and improve consistency',
                'actions': [
                    'Identify automation opportunities',
                    'Develop automation scripts',
                    'Implement workflow automation'
                ],
                'timeline': '60_days',
                'expected_impact': 'medium'
            }
        ])
        
        return sorted(recommendations, key=lambda x: {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}[x['priority']])
    
    def _determine_security_status(self, kpi_values: Dict) -> str:
        """Determine overall security status"""
        security_kpis = [kpi for kpi in kpi_values.values() if kpi['category'] == 'security']
        
        if not security_kpis:
            return 'unknown'
        
        below_target = sum(1 for kpi in security_kpis if kpi['status'] == 'below')
        
        if below_target == 0:
            return 'excellent'
        elif below_target <= len(security_kpis) * 0.2:
            return 'good'
        elif below_target <= len(security_kpis) * 0.5:
            return 'acceptable'
        else:
            return 'needs_attention'
    
    def _identify_achievements(self, kpi_values: Dict) -> List[str]:
        """Identify key achievements"""
        achievements = []
        
        for kpi_id, kpi_data in kpi_values.items():
            if kpi_data['status'] == 'exceeding':
                achievements.append(f"{kpi_data['name']} exceeding target by {kpi_data['variance_percentage']:.1f}%")
        
        # Add some standard achievements
        achievements.extend([
            "Zero critical security incidents in the last 24 hours",
            "Threat detection accuracy improved by 3% this month",
            "System uptime maintained at 99.97%"
        ])
        
        return achievements[:5]  # Return top 5
    
    def _identify_critical_issues(self, kpi_values: Dict, risk_assessment: Dict) -> List[str]:
        """Identify critical issues requiring attention"""
        issues = []
        
        # KPI-based issues
        for kpi_id, kpi_data in kpi_values.items():
            if kpi_data['status'] == 'below' and kpi_data['variance_percentage'] < -10:
                issues.append(f"{kpi_data['name']} significantly below target ({kpi_data['variance_percentage']:.1f}%)")
        
        # Risk-based issues
        for risk in risk_assessment['top_risks']:
            if risk['risk_score'] > 7.5:
                issues.append(f"High-risk threat: {risk['description']}")
        
        return issues
    
    def _assess_business_impact(self) -> Dict[str, Any]:
        """Assess business impact of security operations"""
        return {
            'operational_efficiency': {
                'score': 92.5,
                'trend': 'improving',
                'factors': ['Reduced false positives', 'Faster response times']
            },
            'cost_effectiveness': {
                'score': 88.3,
                'trend': 'stable',
                'factors': ['Automation savings', 'Reduced incident costs']
            },
            'user_productivity': {
                'score': 94.1,
                'trend': 'stable',
                'factors': ['Minimal security friction', 'Fast authentication']
            },
            'reputation_protection': {
                'score': 96.8,
                'trend': 'improving',
                'factors': ['No data breaches', 'Proactive threat response']
            }
        }
    
    def _get_compliance_status(self) -> Dict[str, Any]:
        """Get compliance status across frameworks"""
        return {
            'overall_score': 98.5,
            'frameworks': {
                'ISO_27001': {'score': 99.2, 'status': 'compliant', 'last_audit': '2024-01-15'},
                'SOC_2': {'score': 98.8, 'status': 'compliant', 'last_audit': '2024-02-01'},
                'GDPR': {'score': 97.5, 'status': 'compliant', 'last_audit': '2024-01-20'},
                'HIPAA': {'score': 98.9, 'status': 'compliant', 'last_audit': '2024-01-10'},
                'PCI_DSS': {'score': 99.1, 'status': 'compliant', 'last_audit': '2024-01-25'}
            },
            'upcoming_audits': [
                {'framework': 'ISO_27001', 'date': '2024-07-15', 'type': 'annual'},
                {'framework': 'SOC_2', 'date': '2024-08-01', 'type': 'quarterly'}
            ],
            'action_items': [
                'Update privacy policy documentation',
                'Complete staff security training',
                'Review access control policies'
            ]
        }
    
    def _get_resource_utilization(self) -> Dict[str, Any]:
        """Get resource utilization metrics"""
        return {
            'compute_resources': {
                'cpu_utilization': 68.5,
                'memory_utilization': 72.3,
                'storage_utilization': 45.8,
                'network_utilization': 34.2
            },
            'security_team': {
                'analysts_available': 8,
                'analysts_on_duty': 3,
                'workload_distribution': 'balanced',
                'overtime_hours': 12.5
            },
            'budget_utilization': {
                'annual_budget': 2500000,
                'spent_to_date': 1875000,
                'remaining_budget': 625000,
                'utilization_percentage': 75.0
            }
        }
    
    def _get_cost_analysis(self) -> Dict[str, Any]:
        """Get cost analysis and ROI metrics"""
        return {
            'total_security_investment': 2500000,
            'cost_breakdown': {
                'personnel': 1500000,
                'technology': 750000,
                'training': 150000,
                'compliance': 100000
            },
            'roi_metrics': {
                'incidents_prevented': 45,
                'estimated_loss_prevented': 15000000,
                'roi_percentage': 600.0,
                'payback_period_months': 8.5
            },
            'cost_per_metric': {
                'cost_per_threat_detected': 125.50,
                'cost_per_incident_prevented': 55555.56,
                'cost_per_user_protected': 25.00
            }
        }

class DashboardManager:
    """Main dashboard management class"""
    
    def __init__(self):
        self.analytics = RealTimeAnalytics()
        self.executive_reporting = ExecutiveReporting(self.analytics)
        self.dashboard_config = self._load_dashboard_config()
        self.active_sessions = {}
        
        # Initialize with sample data
        self._populate_sample_data()
    
    def _load_dashboard_config(self) -> Dict[str, Any]:
        """Load dashboard configuration"""
        return {
            'refresh_interval': 30,  # seconds
            'data_retention_days': 90,
            'alert_thresholds': {
                'critical_alerts': 5,
                'threat_score': 8.0,
                'system_health': 85.0
            },
            'widgets': [
                'threat_overview',
                'security_metrics',
                'system_health',
                'recent_alerts',
                'threat_map',
                'performance_charts'
            ],
            'user_preferences': {
                'theme': 'dark',
                'timezone': 'UTC',
                'notifications': True
            }
        }
    
    def _populate_sample_data(self):
        """Populate with sample data for demonstration"""
        current_time = datetime.utcnow()
        
        # Add sample metrics
        sample_metrics = [
            DashboardMetric(
                metric_id='threats_detected',
                name='Threats Detected',
                value=23,
                unit='count',
                trend='up',
                change_percentage=15.2,
                timestamp=current_time,
                category='security',
                severity='medium',
                threshold_status='warning'
            ),
            DashboardMetric(
                metric_id='system_cpu',
                name='CPU Utilization',
                value=68.5,
                unit='%',
                trend='stable',
                change_percentage=2.1,
                timestamp=current_time,
                category='performance',
                severity='low',
                threshold_status='normal'
            )
        ]
        
        for metric in sample_metrics:
            self.analytics.add_metric(metric)
        
        # Add sample alerts
        sample_alerts = [
            SecurityAlert(
                alert_id='ALT-001',
                title='Suspicious Network Activity',
                description='Unusual outbound traffic detected from internal host',
                severity='high',
                category='network',
                timestamp=current_time - timedelta(minutes=15),
                source='network_monitor',
                affected_assets=['192.168.1.100'],
                status='investigating',
                assigned_to='analyst_1'
            ),
            SecurityAlert(
                alert_id='ALT-002',
                title='Failed Login Attempts',
                description='Multiple failed login attempts from external IP',
                severity='medium',
                category='authentication',
                timestamp=current_time - timedelta(minutes=30),
                source='auth_system',
                affected_assets=['login_server'],
                status='new',
                assigned_to=None
            )
        ]
        
        for alert in sample_alerts:
            self.analytics.add_alert(alert)
    
    async def get_dashboard_data(self, user_id: str, dashboard_type: str = 'operational') -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        try:
            if dashboard_type == 'executive':
                return await self._get_executive_dashboard(user_id)
            elif dashboard_type == 'analyst':
                return await self._get_analyst_dashboard(user_id)
            else:
                return await self._get_operational_dashboard(user_id)
                
        except Exception as e:
            logger.error(f"Dashboard data retrieval failed: {e}")
            return {'error': str(e)}
    
    async def _get_operational_dashboard(self, user_id: str) -> Dict[str, Any]:
        """Get operational dashboard data"""
        real_time_metrics = self.analytics.get_real_time_metrics()
        
        return {
            'dashboard_type': 'operational',
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'real_time_metrics': real_time_metrics,
            'recent_alerts': [asdict(alert) for alert in list(self.analytics.alert_buffer)[-10:]],
            'threat_indicators': [asdict(threat) for threat in list(self.analytics.threat_buffer)[-20:]],
            'time_series_data': {
                name: list(data)[-48:] for name, data in self.analytics.time_series_data.items()
            },
            'system_status': {
                'services': [
                    {'name': 'Threat Detection', 'status': 'operational', 'uptime': '99.97%'},
                    {'name': 'Log Analysis', 'status': 'operational', 'uptime': '99.95%'},
                    {'name': 'Alert System', 'status': 'operational', 'uptime': '100.00%'},
                    {'name': 'Dashboard', 'status': 'operational', 'uptime': '99.98%'}
                ]
            },
            'quick_actions': [
                {'action': 'investigate_alert', 'label': 'Investigate Top Alert'},
                {'action': 'run_threat_hunt', 'label': 'Run Threat Hunt'},
                {'action': 'update_signatures', 'label': 'Update Signatures'},
                {'action': 'generate_report', 'label': 'Generate Report'}
            ]
        }
    
    async def _get_executive_dashboard(self, user_id: str) -> Dict[str, Any]:
        """Get executive dashboard data"""
        executive_summary = self.executive_reporting.generate_executive_summary()
        real_time_metrics = self.analytics.get_real_time_metrics()
        
        return {
            'dashboard_type': 'executive',
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'executive_summary': executive_summary,
            'key_metrics': {
                'security_score': real_time_metrics['security_score'],
                'active_threats': real_time_metrics['active_threats'],
                'system_health': real_time_metrics['system_health'],
                'uptime': real_time_metrics['uptime_percentage']
            },
            'business_impact': executive_summary['executive_summary']['business_impact'],
            'strategic_recommendations': executive_summary['recommendations'][:3],
            'compliance_summary': executive_summary['compliance_status'],
            'cost_summary': executive_summary['cost_analysis']
        }
    
    async def _get_analyst_dashboard(self, user_id: str) -> Dict[str, Any]:
        """Get analyst dashboard data"""
        real_time_metrics = self.analytics.get_real_time_metrics()
        
        return {
            'dashboard_type': 'analyst',
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'real_time_metrics': real_time_metrics,
            'detailed_alerts': [asdict(alert) for alert in self.analytics.alert_buffer],
            'threat_analysis': {
                'active_campaigns': 3,
                'threat_actors': 7,
                'iocs_tracked': len(self.analytics.threat_buffer),
                'correlation_results': []
            },
            'investigation_tools': [
                {'tool': 'threat_hunter', 'status': 'available'},
                {'tool': 'log_analyzer', 'status': 'available'},
                {'tool': 'network_tracer', 'status': 'available'},
                {'tool': 'malware_sandbox', 'status': 'available'}
            ],
            'workload': {
                'assigned_alerts': 5,
                'pending_investigations': 2,
                'completed_today': 8,
                'escalated_cases': 1
            }
        }

# Global dashboard manager instance
dashboard_manager = DashboardManager() 