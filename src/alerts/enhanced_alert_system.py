#!/usr/bin/env python3
"""
AADN Enhanced Alert System
Advanced alerting with real-time notifications, correlation, and intelligent filtering
"""

import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import queue
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import logging

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    THREAT_DETECTED = "threat_detected"
    BRUTE_FORCE = "brute_force"
    RECONNAISSANCE = "reconnaissance"
    MALWARE = "malware"
    SYSTEM_ANOMALY = "system_anomaly"
    DECOY_INTERACTION = "decoy_interaction"
    PERFORMANCE_ISSUE = "performance_issue"
    SECURITY_BREACH = "security_breach"

class AlertStatus(Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

@dataclass
class Alert:
    id: str
    type: AlertType
    severity: AlertSeverity
    title: str
    description: str
    source_ip: Optional[str]
    target_decoy: Optional[str]
    timestamp: datetime
    status: AlertStatus = AlertStatus.ACTIVE
    metadata: Dict[str, Any] = None
    correlation_id: Optional[str] = None
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        data = asdict(self)
        data['type'] = self.type.value
        data['severity'] = self.severity.value
        data['status'] = self.status.value
        data['timestamp'] = self.timestamp.isoformat()
        if self.acknowledged_at:
            data['acknowledged_at'] = self.acknowledged_at.isoformat()
        if self.resolved_at:
            data['resolved_at'] = self.resolved_at.isoformat()
        return data

class AlertCorrelator:
    """Correlates related alerts to reduce noise"""
    
    def __init__(self, correlation_window: int = 300):  # 5 minutes
        self.correlation_window = correlation_window
        self.correlation_rules = {
            AlertType.BRUTE_FORCE: self._correlate_brute_force,
            AlertType.RECONNAISSANCE: self._correlate_reconnaissance,
            AlertType.THREAT_DETECTED: self._correlate_threats
        }
    
    def correlate_alert(self, alert: Alert, recent_alerts: List[Alert]) -> Optional[str]:
        """Find correlation ID for alert"""
        if alert.type in self.correlation_rules:
            return self.correlation_rules[alert.type](alert, recent_alerts)
        return None
    
    def _correlate_brute_force(self, alert: Alert, recent_alerts: List[Alert]) -> Optional[str]:
        """Correlate brute force attacks from same IP"""
        cutoff_time = alert.timestamp - timedelta(seconds=self.correlation_window)
        
        for recent_alert in recent_alerts:
            if (recent_alert.type == AlertType.BRUTE_FORCE and
                recent_alert.source_ip == alert.source_ip and
                recent_alert.timestamp > cutoff_time):
                return recent_alert.correlation_id or recent_alert.id
        return None
    
    def _correlate_reconnaissance(self, alert: Alert, recent_alerts: List[Alert]) -> Optional[str]:
        """Correlate reconnaissance activities"""
        cutoff_time = alert.timestamp - timedelta(seconds=self.correlation_window * 2)  # Longer window
        
        for recent_alert in recent_alerts:
            if (recent_alert.type == AlertType.RECONNAISSANCE and
                recent_alert.source_ip == alert.source_ip and
                recent_alert.timestamp > cutoff_time):
                return recent_alert.correlation_id or recent_alert.id
        return None
    
    def _correlate_threats(self, alert: Alert, recent_alerts: List[Alert]) -> Optional[str]:
        """Correlate general threats"""
        cutoff_time = alert.timestamp - timedelta(seconds=self.correlation_window)
        
        for recent_alert in recent_alerts:
            if (recent_alert.type == AlertType.THREAT_DETECTED and
                recent_alert.source_ip == alert.source_ip and
                recent_alert.timestamp > cutoff_time):
                return recent_alert.correlation_id or recent_alert.id
        return None

class NotificationChannel:
    """Base class for notification channels"""
    
    def send(self, alert: Alert) -> bool:
        raise NotImplementedError

class EmailNotification(NotificationChannel):
    """Email notification channel"""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str, recipients: List[str]):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.recipients = recipients
    
    def send(self, alert: Alert) -> bool:
        """Send email notification"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['To'] = ', '.join(self.recipients)
            msg['Subject'] = f"AADN Alert: {alert.severity.value.upper()} - {alert.title}"
            
            body = f"""
            Alert Details:
            
            Severity: {alert.severity.value.upper()}
            Type: {alert.type.value}
            Title: {alert.title}
            Description: {alert.description}
            Source IP: {alert.source_ip or 'Unknown'}
            Target Decoy: {alert.target_decoy or 'N/A'}
            Timestamp: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
            
            Alert ID: {alert.id}
            
            Please review and take appropriate action.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()
            
            return True
        except Exception as e:
            logging.error(f"Failed to send email notification: {e}")
            return False

class SlackNotification(NotificationChannel):
    """Slack notification channel"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def send(self, alert: Alert) -> bool:
        """Send Slack notification"""
        try:
            color_map = {
                AlertSeverity.LOW: "#36a64f",
                AlertSeverity.MEDIUM: "#ff9500",
                AlertSeverity.HIGH: "#ff0000",
                AlertSeverity.CRITICAL: "#8b0000"
            }
            
            payload = {
                "attachments": [{
                    "color": color_map.get(alert.severity, "#36a64f"),
                    "title": f"AADN Alert: {alert.title}",
                    "fields": [
                        {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                        {"title": "Type", "value": alert.type.value, "short": True},
                        {"title": "Source IP", "value": alert.source_ip or "Unknown", "short": True},
                        {"title": "Target", "value": alert.target_decoy or "N/A", "short": True},
                        {"title": "Description", "value": alert.description, "short": False}
                    ],
                    "footer": "AADN Security System",
                    "ts": int(alert.timestamp.timestamp())
                }]
            }
            
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logging.error(f"Failed to send Slack notification: {e}")
            return False

class WebhookNotification(NotificationChannel):
    """Generic webhook notification"""
    
    def __init__(self, webhook_url: str, headers: Dict[str, str] = None):
        self.webhook_url = webhook_url
        self.headers = headers or {}
    
    def send(self, alert: Alert) -> bool:
        """Send webhook notification"""
        try:
            payload = alert.to_dict()
            response = requests.post(
                self.webhook_url, 
                json=payload, 
                headers=self.headers,
                timeout=10
            )
            return response.status_code in [200, 201, 202]
        except Exception as e:
            logging.error(f"Failed to send webhook notification: {e}")
            return False

class EnhancedAlertSystem:
    """Enhanced alert system with correlation and notifications"""
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.correlator = AlertCorrelator()
        self.notification_channels: List[NotificationChannel] = []
        self.alert_queue = queue.Queue()
        self.suppression_rules: Dict[str, Dict] = {}
        self.escalation_rules: Dict[AlertSeverity, int] = {
            AlertSeverity.LOW: 3600,      # 1 hour
            AlertSeverity.MEDIUM: 1800,   # 30 minutes
            AlertSeverity.HIGH: 600,      # 10 minutes
            AlertSeverity.CRITICAL: 60    # 1 minute
        }
        
        # Start background processing
        self._start_processing_thread()
    
    def _start_processing_thread(self):
        """Start background thread for alert processing"""
        def process_alerts():
            while True:
                try:
                    alert = self.alert_queue.get(timeout=1)
                    self._process_alert(alert)
                    self.alert_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    logging.error(f"Error processing alert: {e}")
        
        processing_thread = threading.Thread(target=process_alerts, daemon=True)
        processing_thread.start()
    
    def add_notification_channel(self, channel: NotificationChannel):
        """Add notification channel"""
        self.notification_channels.append(channel)
    
    def create_alert(self, 
                    alert_type: AlertType,
                    severity: AlertSeverity,
                    title: str,
                    description: str,
                    source_ip: Optional[str] = None,
                    target_decoy: Optional[str] = None,
                    metadata: Dict[str, Any] = None) -> Alert:
        """Create new alert"""
        
        alert = Alert(
            id=str(uuid.uuid4()),
            type=alert_type,
            severity=severity,
            title=title,
            description=description,
            source_ip=source_ip,
            target_decoy=target_decoy,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        # Check for correlation
        recent_alerts = self._get_recent_alerts(hours=1)
        correlation_id = self.correlator.correlate_alert(alert, recent_alerts)
        if correlation_id:
            alert.correlation_id = correlation_id
        
        # Check suppression rules
        if not self._is_suppressed(alert):
            self.alerts.append(alert)
            self.alert_queue.put(alert)
        
        return alert
    
    def _process_alert(self, alert: Alert):
        """Process alert (send notifications, etc.)"""
        # Send notifications based on severity
        if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            self._send_notifications(alert)
        elif alert.severity == AlertSeverity.MEDIUM:
            # Send notifications for medium alerts if not correlated
            if not alert.correlation_id:
                self._send_notifications(alert)
    
    def _send_notifications(self, alert: Alert):
        """Send notifications through all channels"""
        for channel in self.notification_channels:
            try:
                channel.send(alert)
            except Exception as e:
                logging.error(f"Failed to send notification: {e}")
    
    def _get_recent_alerts(self, hours: int = 1) -> List[Alert]:
        """Get recent alerts for correlation"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [alert for alert in self.alerts if alert.timestamp > cutoff_time]
    
    def _is_suppressed(self, alert: Alert) -> bool:
        """Check if alert should be suppressed"""
        # Implement suppression logic based on rules
        for rule_name, rule in self.suppression_rules.items():
            if self._matches_suppression_rule(alert, rule):
                return True
        return False
    
    def _matches_suppression_rule(self, alert: Alert, rule: Dict) -> bool:
        """Check if alert matches suppression rule"""
        # Simple rule matching - can be extended
        if 'type' in rule and alert.type.value != rule['type']:
            return False
        if 'source_ip' in rule and alert.source_ip != rule['source_ip']:
            return False
        if 'severity' in rule and alert.severity.value != rule['severity']:
            return False
        return True
    
    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = user
                alert.acknowledged_at = datetime.utcnow()
                return True
        return False
    
    def resolve_alert(self, alert_id: str, user: str) -> bool:
        """Resolve an alert"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.utcnow()
                return True
        return False
    
    def get_alerts(self, 
                  status: Optional[AlertStatus] = None,
                  severity: Optional[AlertSeverity] = None,
                  limit: int = 100) -> List[Dict[str, Any]]:
        """Get alerts with filtering"""
        filtered_alerts = self.alerts
        
        if status:
            filtered_alerts = [a for a in filtered_alerts if a.status == status]
        if severity:
            filtered_alerts = [a for a in filtered_alerts if a.severity == severity]
        
        # Sort by timestamp (newest first)
        filtered_alerts.sort(key=lambda x: x.timestamp, reverse=True)
        
        return [alert.to_dict() for alert in filtered_alerts[:limit]]
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics"""
        now = datetime.utcnow()
        last_24h = now - timedelta(hours=24)
        last_hour = now - timedelta(hours=1)
        
        recent_alerts = [a for a in self.alerts if a.timestamp > last_24h]
        hourly_alerts = [a for a in self.alerts if a.timestamp > last_hour]
        
        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = len([
                a for a in recent_alerts if a.severity == severity
            ])
        
        type_counts = {}
        for alert_type in AlertType:
            type_counts[alert_type.value] = len([
                a for a in recent_alerts if a.type == alert_type
            ])
        
        return {
            "total_alerts": len(self.alerts),
            "alerts_24h": len(recent_alerts),
            "alerts_1h": len(hourly_alerts),
            "active_alerts": len([a for a in self.alerts if a.status == AlertStatus.ACTIVE]),
            "severity_distribution": severity_counts,
            "type_distribution": type_counts,
            "avg_resolution_time": self._calculate_avg_resolution_time()
        }
    
    def _calculate_avg_resolution_time(self) -> Optional[float]:
        """Calculate average resolution time in minutes"""
        resolved_alerts = [
            a for a in self.alerts 
            if a.status == AlertStatus.RESOLVED and a.resolved_at
        ]
        
        if not resolved_alerts:
            return None
        
        total_time = sum([
            (alert.resolved_at - alert.timestamp).total_seconds()
            for alert in resolved_alerts
        ])
        
        return total_time / len(resolved_alerts) / 60  # Convert to minutes
    
    def add_suppression_rule(self, name: str, rule: Dict[str, Any]):
        """Add alert suppression rule"""
        self.suppression_rules[name] = rule
    
    def remove_suppression_rule(self, name: str) -> bool:
        """Remove suppression rule"""
        if name in self.suppression_rules:
            del self.suppression_rules[name]
            return True
        return False

# Global alert system instance
alert_system = EnhancedAlertSystem()

# Convenience functions
def create_threat_alert(source_ip: str, threat_type: str, severity: AlertSeverity, details: str):
    """Create a threat detection alert"""
    return alert_system.create_alert(
        alert_type=AlertType.THREAT_DETECTED,
        severity=severity,
        title=f"{threat_type} detected from {source_ip}",
        description=details,
        source_ip=source_ip,
        metadata={"threat_type": threat_type}
    )

def create_brute_force_alert(source_ip: str, target_decoy: str, attempt_count: int):
    """Create a brute force alert"""
    severity = AlertSeverity.HIGH if attempt_count > 10 else AlertSeverity.MEDIUM
    return alert_system.create_alert(
        alert_type=AlertType.BRUTE_FORCE,
        severity=severity,
        title=f"Brute force attack from {source_ip}",
        description=f"Detected {attempt_count} failed login attempts on {target_decoy}",
        source_ip=source_ip,
        target_decoy=target_decoy,
        metadata={"attempt_count": attempt_count}
    )

def create_reconnaissance_alert(source_ip: str, scan_type: str, ports_scanned: List[int]):
    """Create a reconnaissance alert"""
    return alert_system.create_alert(
        alert_type=AlertType.RECONNAISSANCE,
        severity=AlertSeverity.MEDIUM,
        title=f"Reconnaissance activity from {source_ip}",
        description=f"{scan_type} scan detected targeting {len(ports_scanned)} ports",
        source_ip=source_ip,
        metadata={"scan_type": scan_type, "ports": ports_scanned}
    ) 