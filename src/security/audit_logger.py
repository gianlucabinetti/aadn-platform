#!/usr/bin/env python3
"""
Enterprise Audit Logging System
Comprehensive security event logging for compliance and monitoring
"""

import json
import time
import hashlib
import logging
import logging.handlers
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timezone
from pathlib import Path
import gzip
import threading
from collections import deque, defaultdict
import uuid

class EventSeverity(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class EventCategory(Enum):
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    SYSTEM_CHANGE = "system_change"
    SECURITY_EVENT = "security_event"
    THREAT_DETECTION = "threat_detection"
    COMPLIANCE = "compliance"
    PERFORMANCE = "performance"
    ERROR = "error"

class ComplianceFramework(Enum):
    GDPR = "gdpr"
    SOC2 = "soc2"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    ISO27001 = "iso27001"
    NIST = "nist"

@dataclass
class AuditEvent:
    event_id: str
    timestamp: datetime
    severity: EventSeverity
    category: EventCategory
    event_type: str
    source_ip: str
    user_id: Optional[str]
    session_id: Optional[str]
    resource: str
    action: str
    outcome: str  # success, failure, error
    details: Dict[str, Any]
    compliance_tags: List[ComplianceFramework]
    risk_score: int  # 0-100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity.value,
            'category': self.category.value,
            'event_type': self.event_type,
            'source_ip': self.source_ip,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'resource': self.resource,
            'action': self.action,
            'outcome': self.outcome,
            'details': self.details,
            'compliance_tags': [tag.value for tag in self.compliance_tags],
            'risk_score': self.risk_score
        }

class EnterpriseAuditLogger:
    def __init__(self, log_directory: str = "logs/audit"):
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Event buffer for batch processing
        self.event_buffer = deque(maxlen=1000)
        self.buffer_lock = threading.Lock()
        
        # Real-time statistics
        self.stats = {
            'total_events': 0,
            'events_by_severity': defaultdict(int),
            'events_by_category': defaultdict(int),
            'high_risk_events': 0,
            'compliance_events': defaultdict(int)
        }
        
        # Configure logging
        self._setup_logging()
        
        # Start background processing
        self._start_background_processor()
        
        # Compliance rules
        self.compliance_rules = self._load_compliance_rules()
        
        # Risk scoring rules
        self.risk_rules = self._load_risk_rules()
        
    def _setup_logging(self):
        """Setup structured logging configuration"""
        # Main audit logger
        self.audit_logger = logging.getLogger('audit')
        self.audit_logger.setLevel(logging.INFO)
        
        # Security events logger
        self.security_logger = logging.getLogger('security')
        self.security_logger.setLevel(logging.WARNING)
        
        # Compliance logger
        self.compliance_logger = logging.getLogger('compliance')
        self.compliance_logger.setLevel(logging.INFO)
        
        # Create formatters
        audit_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S UTC'
        )
        
        # File handlers with rotation
        audit_handler = logging.handlers.RotatingFileHandler(
            self.log_directory / 'audit.log',
            maxBytes=100*1024*1024,  # 100MB
            backupCount=10
        )
        audit_handler.setFormatter(audit_formatter)
        self.audit_logger.addHandler(audit_handler)
        
        security_handler = logging.handlers.RotatingFileHandler(
            self.log_directory / 'security.log',
            maxBytes=50*1024*1024,  # 50MB
            backupCount=20
        )
        security_handler.setFormatter(audit_formatter)
        self.security_logger.addHandler(security_handler)
        
        compliance_handler = logging.handlers.RotatingFileHandler(
            self.log_directory / 'compliance.log',
            maxBytes=50*1024*1024,  # 50MB
            backupCount=30
        )
        compliance_handler.setFormatter(audit_formatter)
        self.compliance_logger.addHandler(compliance_handler)
    
    def _load_compliance_rules(self) -> Dict[ComplianceFramework, Dict]:
        """Load compliance framework requirements"""
        return {
            ComplianceFramework.GDPR: {
                'required_events': [
                    'data_access', 'data_modification', 'data_deletion',
                    'consent_given', 'consent_withdrawn', 'data_export'
                ],
                'retention_period': 2555,  # 7 years in days
                'anonymization_required': True
            },
            ComplianceFramework.SOC2: {
                'required_events': [
                    'authentication', 'authorization', 'system_change',
                    'data_access', 'security_event'
                ],
                'retention_period': 2555,  # 7 years
                'integrity_verification': True
            },
            ComplianceFramework.HIPAA: {
                'required_events': [
                    'phi_access', 'phi_modification', 'authentication',
                    'authorization', 'security_event'
                ],
                'retention_period': 2190,  # 6 years
                'encryption_required': True
            },
            ComplianceFramework.PCI_DSS: {
                'required_events': [
                    'cardholder_data_access', 'authentication',
                    'authorization', 'system_change'
                ],
                'retention_period': 365,  # 1 year
                'real_time_monitoring': True
            }
        }
    
    def _load_risk_rules(self) -> Dict[str, int]:
        """Load risk scoring rules"""
        return {
            # Authentication events
            'login_success': 1,
            'login_failure': 5,
            'multiple_login_failures': 20,
            'privilege_escalation': 30,
            'admin_login': 10,
            
            # Data access events
            'data_read': 2,
            'data_write': 5,
            'data_delete': 15,
            'bulk_data_access': 25,
            'sensitive_data_access': 20,
            
            # Security events
            'threat_detected': 40,
            'malware_detected': 60,
            'intrusion_attempt': 50,
            'ddos_attack': 45,
            'sql_injection': 55,
            
            # System events
            'configuration_change': 10,
            'user_creation': 8,
            'user_deletion': 12,
            'permission_change': 15,
            
            # Compliance events
            'compliance_violation': 35,
            'audit_log_tampering': 80,
            'unauthorized_access': 45
        }
    
    def _start_background_processor(self):
        """Start background thread for processing events"""
        def process_events():
            while True:
                try:
                    self._process_event_buffer()
                    time.sleep(1)  # Process every second
                except Exception as e:
                    print(f"Error in background processor: {e}")
        
        processor_thread = threading.Thread(target=process_events, daemon=True)
        processor_thread.start()
    
    def _process_event_buffer(self):
        """Process events from buffer"""
        events_to_process = []
        
        with self.buffer_lock:
            while self.event_buffer and len(events_to_process) < 100:
                events_to_process.append(self.event_buffer.popleft())
        
        for event in events_to_process:
            self._write_event_to_logs(event)
            self._update_statistics(event)
            self._check_compliance_requirements(event)
            self._check_security_alerts(event)
    
    def _write_event_to_logs(self, event: AuditEvent):
        """Write event to appropriate log files"""
        event_json = json.dumps(event.to_dict())
        
        # Write to main audit log
        self.audit_logger.info(event_json)
        
        # Write to security log if security-related
        if event.category in [EventCategory.SECURITY_EVENT, EventCategory.THREAT_DETECTION]:
            self.security_logger.warning(event_json)
        
        # Write to compliance log if compliance-tagged
        if event.compliance_tags:
            self.compliance_logger.info(event_json)
        
        # Write high-risk events to separate file
        if event.risk_score >= 50:
            high_risk_file = self.log_directory / f"high_risk_{datetime.now().strftime('%Y%m%d')}.log"
            with open(high_risk_file, 'a') as f:
                f.write(f"{event_json}\n")
    
    def _update_statistics(self, event: AuditEvent):
        """Update real-time statistics"""
        self.stats['total_events'] += 1
        self.stats['events_by_severity'][event.severity.value] += 1
        self.stats['events_by_category'][event.category.value] += 1
        
        if event.risk_score >= 50:
            self.stats['high_risk_events'] += 1
        
        for tag in event.compliance_tags:
            self.stats['compliance_events'][tag.value] += 1
    
    def _check_compliance_requirements(self, event: AuditEvent):
        """Check if event meets compliance requirements"""
        for framework in event.compliance_tags:
            rules = self.compliance_rules.get(framework, {})
            
            # Check if event type is required for this framework
            if event.event_type in rules.get('required_events', []):
                # Log compliance event
                compliance_event = AuditEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc),
                    severity=EventSeverity.INFO,
                    category=EventCategory.COMPLIANCE,
                    event_type='compliance_requirement_met',
                    source_ip=event.source_ip,
                    user_id=event.user_id,
                    session_id=event.session_id,
                    resource=f"compliance_{framework.value}",
                    action='requirement_check',
                    outcome='success',
                    details={
                        'framework': framework.value,
                        'original_event_id': event.event_id,
                        'requirement_type': event.event_type
                    },
                    compliance_tags=[framework],
                    risk_score=5
                )
                
                with self.buffer_lock:
                    self.event_buffer.append(compliance_event)
    
    def _check_security_alerts(self, event: AuditEvent):
        """Check if event should trigger security alerts"""
        alert_conditions = [
            (event.risk_score >= 70, "critical_risk_event"),
            (event.category == EventCategory.THREAT_DETECTION, "threat_detected"),
            (event.event_type == "multiple_login_failures", "brute_force_attempt"),
            (event.event_type == "privilege_escalation", "privilege_escalation_attempt"),
            (event.outcome == "failure" and event.risk_score >= 30, "suspicious_failure")
        ]
        
        for condition, alert_type in alert_conditions:
            if condition:
                self._create_security_alert(event, alert_type)
    
    def _create_security_alert(self, original_event: AuditEvent, alert_type: str):
        """Create security alert based on audit event"""
        alert_event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            severity=EventSeverity.CRITICAL,
            category=EventCategory.SECURITY_EVENT,
            event_type=f"security_alert_{alert_type}",
            source_ip=original_event.source_ip,
            user_id=original_event.user_id,
            session_id=original_event.session_id,
            resource="security_monitoring",
            action="alert_generated",
            outcome="success",
            details={
                'alert_type': alert_type,
                'original_event_id': original_event.event_id,
                'trigger_reason': f"Risk score: {original_event.risk_score}",
                'recommended_action': self._get_recommended_action(alert_type)
            },
            compliance_tags=[ComplianceFramework.SOC2, ComplianceFramework.ISO27001],
            risk_score=80
        )
        
        with self.buffer_lock:
            self.event_buffer.append(alert_event)
    
    def _get_recommended_action(self, alert_type: str) -> str:
        """Get recommended action for alert type"""
        recommendations = {
            "critical_risk_event": "Immediate investigation required",
            "threat_detected": "Block source IP and investigate",
            "brute_force_attempt": "Implement account lockout",
            "privilege_escalation_attempt": "Review user permissions immediately",
            "suspicious_failure": "Monitor user activity closely"
        }
        return recommendations.get(alert_type, "Review and investigate")
    
    def _calculate_risk_score(self, event_type: str, details: Dict[str, Any]) -> int:
        """Calculate risk score for event"""
        base_score = self.risk_rules.get(event_type, 5)
        
        # Adjust based on details
        multipliers = {
            'admin_user': 1.5,
            'sensitive_data': 1.3,
            'external_ip': 1.2,
            'after_hours': 1.1,
            'multiple_attempts': 2.0
        }
        
        final_score = base_score
        for factor, multiplier in multipliers.items():
            if details.get(factor, False):
                final_score *= multiplier
        
        return min(int(final_score), 100)  # Cap at 100
    
    def _determine_compliance_tags(self, event_type: str, resource: str, 
                                 details: Dict[str, Any]) -> List[ComplianceFramework]:
        """Determine which compliance frameworks apply to this event"""
        tags = []
        
        # GDPR - Personal data events
        if any(keyword in resource.lower() for keyword in ['user', 'personal', 'profile', 'contact']):
            tags.append(ComplianceFramework.GDPR)
        
        # SOC2 - Security and availability events
        if event_type in ['authentication', 'authorization', 'system_change', 'security_event']:
            tags.append(ComplianceFramework.SOC2)
        
        # HIPAA - Health information events
        if any(keyword in resource.lower() for keyword in ['health', 'medical', 'patient', 'phi']):
            tags.append(ComplianceFramework.HIPAA)
        
        # PCI DSS - Payment card events
        if any(keyword in resource.lower() for keyword in ['payment', 'card', 'transaction']):
            tags.append(ComplianceFramework.PCI_DSS)
        
        # ISO27001 - Information security events
        if event_type in ['security_event', 'threat_detection', 'system_change']:
            tags.append(ComplianceFramework.ISO27001)
        
        return tags
    
    def log_event(self, event_type: str, source_ip: str, resource: str, action: str,
                  outcome: str, user_id: Optional[str] = None, session_id: Optional[str] = None,
                  details: Optional[Dict[str, Any]] = None, severity: Optional[EventSeverity] = None,
                  category: Optional[EventCategory] = None) -> str:
        """Log an audit event"""
        
        if details is None:
            details = {}
        
        # Auto-determine severity if not provided
        if severity is None:
            if outcome == "failure":
                severity = EventSeverity.WARNING
            elif event_type.startswith("security_") or event_type.startswith("threat_"):
                severity = EventSeverity.ERROR
            else:
                severity = EventSeverity.INFO
        
        # Auto-determine category if not provided
        if category is None:
            category_mapping = {
                'login': EventCategory.AUTHENTICATION,
                'logout': EventCategory.AUTHENTICATION,
                'access': EventCategory.DATA_ACCESS,
                'create': EventCategory.SYSTEM_CHANGE,
                'update': EventCategory.SYSTEM_CHANGE,
                'delete': EventCategory.SYSTEM_CHANGE,
                'threat': EventCategory.THREAT_DETECTION,
                'security': EventCategory.SECURITY_EVENT
            }
            
            for keyword, cat in category_mapping.items():
                if keyword in event_type.lower():
                    category = cat
                    break
            else:
                category = EventCategory.DATA_ACCESS
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(event_type, details)
        
        # Determine compliance tags
        compliance_tags = self._determine_compliance_tags(event_type, resource, details)
        
        # Create audit event
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            severity=severity,
            category=category,
            event_type=event_type,
            source_ip=source_ip,
            user_id=user_id,
            session_id=session_id,
            resource=resource,
            action=action,
            outcome=outcome,
            details=details,
            compliance_tags=compliance_tags,
            risk_score=risk_score
        )
        
        # Add to buffer for processing
        with self.buffer_lock:
            self.event_buffer.append(event)
        
        return event.event_id
    
    def log_authentication_event(self, user_id: str, source_ip: str, outcome: str,
                               session_id: Optional[str] = None, details: Optional[Dict] = None):
        """Log authentication event"""
        event_type = "login_success" if outcome == "success" else "login_failure"
        severity = EventSeverity.INFO if outcome == "success" else EventSeverity.WARNING
        
        return self.log_event(
            event_type=event_type,
            source_ip=source_ip,
            resource="authentication_system",
            action="authenticate",
            outcome=outcome,
            user_id=user_id,
            session_id=session_id,
            details=details or {},
            severity=severity,
            category=EventCategory.AUTHENTICATION
        )
    
    def log_data_access_event(self, user_id: str, source_ip: str, resource: str,
                            action: str, outcome: str, session_id: Optional[str] = None,
                            details: Optional[Dict] = None):
        """Log data access event"""
        return self.log_event(
            event_type="data_access",
            source_ip=source_ip,
            resource=resource,
            action=action,
            outcome=outcome,
            user_id=user_id,
            session_id=session_id,
            details=details or {},
            category=EventCategory.DATA_ACCESS
        )
    
    def log_threat_detection_event(self, source_ip: str, threat_type: str, severity: str,
                                 details: Optional[Dict] = None):
        """Log threat detection event"""
        severity_mapping = {
            'low': EventSeverity.INFO,
            'medium': EventSeverity.WARNING,
            'high': EventSeverity.ERROR,
            'critical': EventSeverity.CRITICAL
        }
        
        return self.log_event(
            event_type=f"threat_detected_{threat_type}",
            source_ip=source_ip,
            resource="threat_detection_system",
            action="detect_threat",
            outcome="success",
            details=details or {},
            severity=severity_mapping.get(severity.lower(), EventSeverity.WARNING),
            category=EventCategory.THREAT_DETECTION
        )
    
    def get_audit_statistics(self) -> Dict[str, Any]:
        """Get audit logging statistics"""
        return {
            **self.stats,
            'buffer_size': len(self.event_buffer),
            'log_files': {
                'audit_log_size': self._get_file_size('audit.log'),
                'security_log_size': self._get_file_size('security.log'),
                'compliance_log_size': self._get_file_size('compliance.log')
            }
        }
    
    def _get_file_size(self, filename: str) -> int:
        """Get file size in bytes"""
        try:
            return (self.log_directory / filename).stat().st_size
        except FileNotFoundError:
            return 0
    
    def search_events(self, start_time: datetime, end_time: datetime,
                     event_type: Optional[str] = None, user_id: Optional[str] = None,
                     source_ip: Optional[str] = None, min_risk_score: int = 0) -> List[Dict]:
        """Search audit events (simplified implementation)"""
        # In production, this would use a proper search index
        events = []
        
        # Read from current log files
        for log_file in ['audit.log', 'security.log', 'compliance.log']:
            try:
                with open(self.log_directory / log_file, 'r') as f:
                    for line in f:
                        try:
                            # Parse log line to extract JSON
                            json_start = line.find('{')
                            if json_start != -1:
                                event_data = json.loads(line[json_start:])
                                event_time = datetime.fromisoformat(event_data['timestamp'].replace('Z', '+00:00'))
                                
                                # Apply filters
                                if start_time <= event_time <= end_time:
                                    if event_type and event_data.get('event_type') != event_type:
                                        continue
                                    if user_id and event_data.get('user_id') != user_id:
                                        continue
                                    if source_ip and event_data.get('source_ip') != source_ip:
                                        continue
                                    if event_data.get('risk_score', 0) < min_risk_score:
                                        continue
                                    
                                    events.append(event_data)
                        except (json.JSONDecodeError, KeyError, ValueError):
                            continue
            except FileNotFoundError:
                continue
        
        return sorted(events, key=lambda x: x['timestamp'], reverse=True)
    
    def generate_compliance_report(self, framework: ComplianceFramework,
                                 start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate compliance report for specific framework"""
        events = self.search_events(start_date, end_date)
        
        # Filter events for this framework
        framework_events = [e for e in events if framework.value in e.get('compliance_tags', [])]
        
        # Generate report
        report = {
            'framework': framework.value,
            'report_period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'total_events': len(framework_events),
            'event_breakdown': defaultdict(int),
            'risk_analysis': {
                'high_risk_events': 0,
                'medium_risk_events': 0,
                'low_risk_events': 0
            },
            'compliance_status': 'compliant',
            'recommendations': []
        }
        
        # Analyze events
        for event in framework_events:
            report['event_breakdown'][event['event_type']] += 1
            
            risk_score = event.get('risk_score', 0)
            if risk_score >= 70:
                report['risk_analysis']['high_risk_events'] += 1
            elif risk_score >= 30:
                report['risk_analysis']['medium_risk_events'] += 1
            else:
                report['risk_analysis']['low_risk_events'] += 1
        
        # Check compliance requirements
        rules = self.compliance_rules.get(framework, {})
        required_events = rules.get('required_events', [])
        
        for required_event in required_events:
            if required_event not in report['event_breakdown']:
                report['compliance_status'] = 'non_compliant'
                report['recommendations'].append(f"Missing required event type: {required_event}")
        
        return dict(report)

# Global audit logger instance
audit_logger = EnterpriseAuditLogger() 