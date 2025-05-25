"""
AADN Alert System
Intelligent alerting and notification system for threat detection
"""

import asyncio
import json
import logging
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from dataclasses import dataclass, asdict

import aiohttp
from jinja2 import Template

from ..core.logging_config import get_intelligence_logger, log_security_event, SecurityEvent
from ..ai.threat_analyzer import ThreatAnalysis, ThreatLevel

logger = get_intelligence_logger()


class AlertChannel(str, Enum):
    """Alert delivery channels"""
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    TEAMS = "teams"
    SYSLOG = "syslog"
    SMS = "sms"


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class AlertRule:
    """Alert rule configuration"""
    id: str
    name: str
    description: str
    enabled: bool
    channels: List[AlertChannel]
    conditions: Dict[str, Any]
    throttle_minutes: int = 5
    severity: AlertSeverity = AlertSeverity.WARNING
    template: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class Alert:
    """Alert instance"""
    id: str
    rule_id: str
    title: str
    message: str
    severity: AlertSeverity
    threat_analysis: Optional[ThreatAnalysis]
    source_data: Dict[str, Any]
    channels: List[AlertChannel]
    created_at: datetime
    sent_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None


class AlertThrottler:
    """Manages alert throttling to prevent spam"""
    
    def __init__(self):
        self.sent_alerts: Dict[str, datetime] = {}
        self.cleanup_interval = timedelta(hours=1)
        self.last_cleanup = datetime.utcnow()
    
    def should_send_alert(self, rule_id: str, throttle_minutes: int) -> bool:
        """Check if alert should be sent based on throttling rules"""
        self._cleanup_old_entries()
        
        throttle_key = rule_id
        last_sent = self.sent_alerts.get(throttle_key)
        
        if not last_sent:
            return True
        
        time_since_last = datetime.utcnow() - last_sent
        return time_since_last >= timedelta(minutes=throttle_minutes)
    
    def record_sent_alert(self, rule_id: str):
        """Record that an alert was sent"""
        self.sent_alerts[rule_id] = datetime.utcnow()
    
    def _cleanup_old_entries(self):
        """Clean up old throttling entries"""
        now = datetime.utcnow()
        if now - self.last_cleanup < self.cleanup_interval:
            return
        
        cutoff = now - timedelta(hours=24)
        self.sent_alerts = {
            key: timestamp for key, timestamp in self.sent_alerts.items()
            if timestamp > cutoff
        }
        self.last_cleanup = now


class EmailNotifier:
    """Email notification handler"""
    
    def __init__(self, smtp_host: str, smtp_port: int, username: str, password: str, use_tls: bool = True):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.use_tls = use_tls
    
    async def send_alert(self, alert: Alert, recipients: List[str]) -> bool:
        """Send alert via email"""
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[AADN Alert] {alert.title}"
            msg['From'] = self.username
            msg['To'] = ', '.join(recipients)
            
            # Create HTML content
            html_content = self._create_html_content(alert)
            html_part = MIMEText(html_content, 'html')
            
            # Create plain text content
            text_content = self._create_text_content(alert)
            text_part = MIMEText(text_content, 'plain')
            
            msg.attach(text_part)
            msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent successfully: {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert {alert.id}: {e}")
            return False
    
    def _create_html_content(self, alert: Alert) -> str:
        """Create HTML email content"""
        template = Template("""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f44336; color: white; padding: 10px; border-radius: 5px; }
                .content { margin: 20px 0; }
                .details { background-color: #f5f5f5; padding: 15px; border-radius: 5px; }
                .threat-info { margin: 10px 0; }
                .recommendations { background-color: #e3f2fd; padding: 10px; border-radius: 5px; }
                .footer { margin-top: 20px; font-size: 12px; color: #666; }
            </style>
        </head>
        <body>
            <div class="header">
                <h2>🚨 AADN Security Alert</h2>
                <p><strong>{{ alert.title }}</strong></p>
            </div>
            
            <div class="content">
                <p>{{ alert.message }}</p>
                
                <div class="details">
                    <h3>Alert Details</h3>
                    <p><strong>Severity:</strong> {{ alert.severity.upper() }}</p>
                    <p><strong>Time:</strong> {{ alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC') }}</p>
                    <p><strong>Alert ID:</strong> {{ alert.id }}</p>
                </div>
                
                {% if alert.threat_analysis %}
                <div class="threat-info">
                    <h3>Threat Analysis</h3>
                    <p><strong>Threat Level:</strong> {{ alert.threat_analysis.level.upper() }}</p>
                    <p><strong>Category:</strong> {{ alert.threat_analysis.category.upper() }}</p>
                    <p><strong>Source IP:</strong> {{ alert.threat_analysis.source_ip }}</p>
                    <p><strong>Confidence:</strong> {{ (alert.threat_analysis.confidence * 100)|round(1) }}%</p>
                    
                    {% if alert.threat_analysis.mitre_techniques %}
                    <p><strong>MITRE ATT&CK Techniques:</strong> {{ alert.threat_analysis.mitre_techniques|join(', ') }}</p>
                    {% endif %}
                </div>
                
                {% if alert.threat_analysis.recommendations %}
                <div class="recommendations">
                    <h3>Recommendations</h3>
                    <ul>
                    {% for rec in alert.threat_analysis.recommendations %}
                        <li>{{ rec }}</li>
                    {% endfor %}
                    </ul>
                </div>
                {% endif %}
                {% endif %}
            </div>
            
            <div class="footer">
                <p>This alert was generated by the Adaptive AI-Driven Deception Network (AADN)</p>
                <p>For more information, please check the AADN dashboard.</p>
            </div>
        </body>
        </html>
        """)
        
        return template.render(alert=alert)
    
    def _create_text_content(self, alert: Alert) -> str:
        """Create plain text email content"""
        content = f"""
AADN Security Alert
==================

{alert.title}

{alert.message}

Alert Details:
- Severity: {alert.severity.upper()}
- Time: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}
- Alert ID: {alert.id}
"""
        
        if alert.threat_analysis:
            content += f"""
Threat Analysis:
- Threat Level: {alert.threat_analysis.level.upper()}
- Category: {alert.threat_analysis.category.upper()}
- Source IP: {alert.threat_analysis.source_ip}
- Confidence: {alert.threat_analysis.confidence * 100:.1f}%
"""
            
            if alert.threat_analysis.mitre_techniques:
                content += f"- MITRE ATT&CK Techniques: {', '.join(alert.threat_analysis.mitre_techniques)}\n"
            
            if alert.threat_analysis.recommendations:
                content += "\nRecommendations:\n"
                for rec in alert.threat_analysis.recommendations:
                    content += f"- {rec}\n"
        
        content += "\n\nThis alert was generated by the Adaptive AI-Driven Deception Network (AADN)"
        
        return content


class WebhookNotifier:
    """Webhook notification handler"""
    
    async def send_alert(self, alert: Alert, webhook_url: str, headers: Optional[Dict[str, str]] = None) -> bool:
        """Send alert via webhook"""
        try:
            payload = {
                'alert_id': alert.id,
                'title': alert.title,
                'message': alert.message,
                'severity': alert.severity,
                'created_at': alert.created_at.isoformat(),
                'source_data': alert.source_data
            }
            
            if alert.threat_analysis:
                payload['threat_analysis'] = {
                    'threat_id': alert.threat_analysis.threat_id,
                    'level': alert.threat_analysis.level,
                    'category': alert.threat_analysis.category,
                    'confidence': alert.threat_analysis.confidence,
                    'source_ip': alert.threat_analysis.source_ip,
                    'target_decoys': alert.threat_analysis.target_decoys,
                    'mitre_techniques': alert.threat_analysis.mitre_techniques,
                    'recommendations': alert.threat_analysis.recommendations
                }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    json=payload,
                    headers=headers or {},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status < 400:
                        logger.info(f"Webhook alert sent successfully: {alert.id}")
                        return True
                    else:
                        logger.error(f"Webhook returned status {response.status} for alert {alert.id}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send webhook alert {alert.id}: {e}")
            return False


class SlackNotifier:
    """Slack notification handler"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert to Slack"""
        try:
            # Create Slack message
            color = self._get_color_for_severity(alert.severity)
            
            attachment = {
                "color": color,
                "title": alert.title,
                "text": alert.message,
                "fields": [
                    {
                        "title": "Severity",
                        "value": alert.severity.upper(),
                        "short": True
                    },
                    {
                        "title": "Time",
                        "value": alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC'),
                        "short": True
                    }
                ],
                "footer": "AADN Alert System",
                "ts": int(alert.created_at.timestamp())
            }
            
            if alert.threat_analysis:
                attachment["fields"].extend([
                    {
                        "title": "Threat Level",
                        "value": alert.threat_analysis.level.upper(),
                        "short": True
                    },
                    {
                        "title": "Source IP",
                        "value": alert.threat_analysis.source_ip,
                        "short": True
                    },
                    {
                        "title": "Category",
                        "value": alert.threat_analysis.category.upper(),
                        "short": True
                    },
                    {
                        "title": "Confidence",
                        "value": f"{alert.threat_analysis.confidence * 100:.1f}%",
                        "short": True
                    }
                ])
            
            payload = {
                "text": f"🚨 AADN Security Alert",
                "attachments": [attachment]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Slack alert sent successfully: {alert.id}")
                        return True
                    else:
                        logger.error(f"Slack returned status {response.status} for alert {alert.id}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send Slack alert {alert.id}: {e}")
            return False
    
    def _get_color_for_severity(self, severity: AlertSeverity) -> str:
        """Get color code for severity level"""
        color_map = {
            AlertSeverity.INFO: "#36a64f",      # Green
            AlertSeverity.WARNING: "#ff9500",   # Orange
            AlertSeverity.CRITICAL: "#ff0000",  # Red
            AlertSeverity.EMERGENCY: "#8b0000"  # Dark Red
        }
        return color_map.get(severity, "#ff9500")


class AlertEngine:
    """Main alert processing engine"""
    
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.throttler = AlertThrottler()
        self.notifiers = {}
        self.alert_history: List[Alert] = []
        
        # Load default rules
        self._load_default_rules()
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule"""
        self.rules[rule.id] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_id: str):
        """Remove an alert rule"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")
    
    def configure_email_notifier(self, smtp_host: str, smtp_port: int, username: str, password: str, use_tls: bool = True):
        """Configure email notifications"""
        self.notifiers[AlertChannel.EMAIL] = EmailNotifier(smtp_host, smtp_port, username, password, use_tls)
    
    def configure_slack_notifier(self, webhook_url: str):
        """Configure Slack notifications"""
        self.notifiers[AlertChannel.SLACK] = SlackNotifier(webhook_url)
    
    def configure_webhook_notifier(self):
        """Configure webhook notifications"""
        self.notifiers[AlertChannel.WEBHOOK] = WebhookNotifier()
    
    async def process_threat_analysis(self, threat_analysis: ThreatAnalysis) -> List[Alert]:
        """Process a threat analysis and generate alerts"""
        alerts = []
        
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            if self._matches_rule(threat_analysis, rule):
                if self.throttler.should_send_alert(rule.id, rule.throttle_minutes):
                    alert = await self._create_alert(threat_analysis, rule)
                    if alert:
                        alerts.append(alert)
                        await self._send_alert(alert)
                        self.throttler.record_sent_alert(rule.id)
                        self.alert_history.append(alert)
        
        return alerts
    
    def _matches_rule(self, threat_analysis: ThreatAnalysis, rule: AlertRule) -> bool:
        """Check if threat analysis matches alert rule conditions"""
        conditions = rule.conditions
        
        # Check threat level
        if 'min_threat_level' in conditions:
            min_level = ThreatLevel(conditions['min_threat_level'])
            threat_levels = [ThreatLevel.LOW, ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL]
            if threat_levels.index(threat_analysis.level) < threat_levels.index(min_level):
                return False
        
        # Check confidence threshold
        if 'min_confidence' in conditions:
            if threat_analysis.confidence < conditions['min_confidence']:
                return False
        
        # Check attack categories
        if 'categories' in conditions:
            if threat_analysis.category not in conditions['categories']:
                return False
        
        # Check source IP patterns
        if 'source_ip_patterns' in conditions:
            patterns = conditions['source_ip_patterns']
            if not any(pattern in threat_analysis.source_ip for pattern in patterns):
                return False
        
        # Check MITRE techniques
        if 'mitre_techniques' in conditions:
            required_techniques = set(conditions['mitre_techniques'])
            threat_techniques = set(threat_analysis.mitre_techniques)
            if not required_techniques.intersection(threat_techniques):
                return False
        
        return True
    
    async def _create_alert(self, threat_analysis: ThreatAnalysis, rule: AlertRule) -> Optional[Alert]:
        """Create an alert from threat analysis and rule"""
        try:
            alert_id = f"alert_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{rule.id}"
            
            # Generate title and message
            title = f"Threat Detected: {threat_analysis.category.upper()} from {threat_analysis.source_ip}"
            message = f"AADN detected a {threat_analysis.level.upper()} level {threat_analysis.category} attack from {threat_analysis.source_ip}"
            
            if rule.template:
                # Use custom template if provided
                template = Template(rule.template)
                message = template.render(threat=threat_analysis)
            
            alert = Alert(
                id=alert_id,
                rule_id=rule.id,
                title=title,
                message=message,
                severity=rule.severity,
                threat_analysis=threat_analysis,
                source_data=asdict(threat_analysis),
                channels=rule.channels,
                created_at=datetime.utcnow(),
                metadata=rule.metadata or {}
            )
            
            return alert
            
        except Exception as e:
            logger.error(f"Failed to create alert for rule {rule.id}: {e}")
            return None
    
    async def _send_alert(self, alert: Alert):
        """Send alert through configured channels"""
        for channel in alert.channels:
            try:
                if channel == AlertChannel.EMAIL and AlertChannel.EMAIL in self.notifiers:
                    # Get email recipients from alert metadata
                    recipients = alert.metadata.get('email_recipients', ['admin@company.com'])
                    await self.notifiers[AlertChannel.EMAIL].send_alert(alert, recipients)
                
                elif channel == AlertChannel.SLACK and AlertChannel.SLACK in self.notifiers:
                    await self.notifiers[AlertChannel.SLACK].send_alert(alert)
                
                elif channel == AlertChannel.WEBHOOK and AlertChannel.WEBHOOK in self.notifiers:
                    webhook_url = alert.metadata.get('webhook_url')
                    if webhook_url:
                        await self.notifiers[AlertChannel.WEBHOOK].send_alert(alert, webhook_url)
                
                # Log security event
                log_security_event(
                    SecurityEvent.ALERT_SENT,
                    {
                        "alert_id": alert.id,
                        "rule_id": alert.rule_id,
                        "channel": channel,
                        "severity": alert.severity,
                        "threat_id": alert.threat_analysis.threat_id if alert.threat_analysis else None
                    },
                    severity="INFO"
                )
                
            except Exception as e:
                logger.error(f"Failed to send alert {alert.id} via {channel}: {e}")
    
    def _load_default_rules(self):
        """Load default alert rules"""
        # High-severity threat rule
        high_threat_rule = AlertRule(
            id="high_threat_detection",
            name="High Threat Detection",
            description="Alert on high-severity threats",
            enabled=True,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            conditions={
                "min_threat_level": "high",
                "min_confidence": 0.7
            },
            throttle_minutes=5,
            severity=AlertSeverity.CRITICAL,
            metadata={"email_recipients": ["security@company.com"]}
        )
        
        # Brute force attack rule
        brute_force_rule = AlertRule(
            id="brute_force_detection",
            name="Brute Force Attack Detection",
            description="Alert on brute force attacks",
            enabled=True,
            channels=[AlertChannel.EMAIL],
            conditions={
                "categories": ["brute_force"],
                "min_confidence": 0.5
            },
            throttle_minutes=10,
            severity=AlertSeverity.WARNING,
            metadata={"email_recipients": ["admin@company.com"]}
        )
        
        # Coordinated attack rule
        coordinated_attack_rule = AlertRule(
            id="coordinated_attack_detection",
            name="Coordinated Attack Detection",
            description="Alert on coordinated attacks from multiple sources",
            enabled=True,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            conditions={
                "categories": ["reconnaissance"],
                "min_confidence": 0.6
            },
            throttle_minutes=15,
            severity=AlertSeverity.CRITICAL,
            metadata={"email_recipients": ["security@company.com", "admin@company.com"]}
        )
        
        self.rules[high_threat_rule.id] = high_threat_rule
        self.rules[brute_force_rule.id] = brute_force_rule
        self.rules[coordinated_attack_rule.id] = coordinated_attack_rule


# Global alert engine instance
alert_engine = AlertEngine()


async def process_threat_alert(threat_analysis: ThreatAnalysis) -> List[Alert]:
    """Process a threat analysis and send alerts"""
    return await alert_engine.process_threat_analysis(threat_analysis)


def configure_email_alerts(smtp_host: str, smtp_port: int, username: str, password: str, use_tls: bool = True):
    """Configure email alert notifications"""
    alert_engine.configure_email_notifier(smtp_host, smtp_port, username, password, use_tls)


def configure_slack_alerts(webhook_url: str):
    """Configure Slack alert notifications"""
    alert_engine.configure_slack_notifier(webhook_url)


def add_alert_rule(rule: AlertRule):
    """Add a custom alert rule"""
    alert_engine.add_rule(rule) 