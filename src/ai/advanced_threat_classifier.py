#!/usr/bin/env python3
"""
Advanced AI Threat Classification System
Next-generation machine learning threat detection
"""

import numpy as np
import json
import time
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import re
import ipaddress
from datetime import datetime, timedelta

class ThreatCategory(Enum):
    RECONNAISSANCE = "reconnaissance"
    LATERAL_MOVEMENT = "lateral_movement"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    PERSISTENCE = "persistence"
    DEFENSE_EVASION = "defense_evasion"
    CREDENTIAL_ACCESS = "credential_access"
    DISCOVERY = "discovery"
    COLLECTION = "collection"
    COMMAND_CONTROL = "command_control"
    EXFILTRATION = "exfiltration"
    IMPACT = "impact"

class ThreatSeverity(Enum):
    INFO = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5

@dataclass
class ThreatFeatures:
    # Network features
    source_ip: str
    destination_port: int
    protocol: str
    packet_size: int
    connection_duration: float
    
    # Behavioral features
    request_frequency: float
    unique_endpoints: int
    failed_attempts: int
    user_agent_entropy: float
    payload_entropy: float
    
    # Temporal features
    time_of_day: int  # 0-23
    day_of_week: int  # 0-6
    request_interval_variance: float
    
    # Content features
    suspicious_keywords: int
    sql_injection_patterns: int
    xss_patterns: int
    command_injection_patterns: int
    
    # MITRE ATT&CK indicators
    mitre_techniques: List[str]
    
    def to_vector(self) -> np.ndarray:
        """Convert features to numerical vector for ML"""
        return np.array([
            self.destination_port,
            self.packet_size,
            self.connection_duration,
            self.request_frequency,
            self.unique_endpoints,
            self.failed_attempts,
            self.user_agent_entropy,
            self.payload_entropy,
            self.time_of_day,
            self.day_of_week,
            self.request_interval_variance,
            self.suspicious_keywords,
            self.sql_injection_patterns,
            self.xss_patterns,
            self.command_injection_patterns,
            len(self.mitre_techniques)
        ])

@dataclass
class ThreatClassification:
    threat_id: str
    category: ThreatCategory
    severity: ThreatSeverity
    confidence: float
    mitre_techniques: List[str]
    indicators: List[str]
    recommended_actions: List[str]
    timestamp: datetime
    source_ip: str
    target_service: str
    
class AdvancedThreatClassifier:
    def __init__(self):
        # Feature extractors
        self.feature_extractors = {
            'network': self._extract_network_features,
            'behavioral': self._extract_behavioral_features,
            'temporal': self._extract_temporal_features,
            'content': self._extract_content_features,
            'mitre': self._extract_mitre_features
        }
        
        # Threat patterns database
        self.threat_patterns = self._load_threat_patterns()
        
        # ML models (simplified - in production use scikit-learn/tensorflow)
        self.models = {
            'anomaly_detector': self._create_anomaly_detector(),
            'classifier': self._create_threat_classifier(),
            'severity_predictor': self._create_severity_predictor()
        }
        
        # Historical data for learning
        self.interaction_history = deque(maxlen=10000)
        self.threat_history = deque(maxlen=1000)
        
        # Real-time statistics
        self.stats = {
            'total_interactions': 0,
            'threats_detected': 0,
            'false_positives': 0,
            'model_accuracy': 0.95
        }
        
        # MITRE ATT&CK technique mapping
        self.mitre_mapping = self._load_mitre_mapping()
        
    def _load_threat_patterns(self) -> Dict[str, Any]:
        """Load known threat patterns and signatures"""
        return {
            'sql_injection': [
                r"(\%27)|(\')|(\-\-)|(\%23)|(#)",
                r"((\%3D)|(=))[^\n]*((\%27)|(\')|(\-\-)|(\%3B)|(;))",
                r"w*((\%27)|(\'))((\%6F)|o|(\%4F))((\%72)|r|(\%52))"
            ],
            'xss_patterns': [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
                r"<iframe[^>]*>.*?</iframe>"
            ],
            'command_injection': [
                r"[;&|`]",
                r"\$\([^)]*\)",
                r"`[^`]*`",
                r"\|\s*(cat|ls|pwd|whoami|id)"
            ],
            'reconnaissance': [
                r"\.\.\/",
                r"\/etc\/passwd",
                r"\/proc\/",
                r"nmap|nikto|dirb|gobuster"
            ],
            'brute_force': [
                r"admin|administrator|root|test",
                r"password|123456|qwerty|admin"
            ]
        }
    
    def _load_mitre_mapping(self) -> Dict[str, Dict]:
        """Load MITRE ATT&CK technique mapping"""
        return {
            'T1595': {
                'name': 'Active Scanning',
                'category': ThreatCategory.RECONNAISSANCE,
                'severity': ThreatSeverity.MEDIUM,
                'indicators': ['port_scan', 'service_enumeration', 'vulnerability_scan']
            },
            'T1110': {
                'name': 'Brute Force',
                'category': ThreatCategory.CREDENTIAL_ACCESS,
                'severity': ThreatSeverity.HIGH,
                'indicators': ['multiple_failed_logins', 'password_spray', 'credential_stuffing']
            },
            'T1190': {
                'name': 'Exploit Public-Facing Application',
                'category': ThreatCategory.PRIVILEGE_ESCALATION,
                'severity': ThreatSeverity.CRITICAL,
                'indicators': ['sql_injection', 'xss', 'command_injection', 'buffer_overflow']
            },
            'T1083': {
                'name': 'File and Directory Discovery',
                'category': ThreatCategory.DISCOVERY,
                'severity': ThreatSeverity.MEDIUM,
                'indicators': ['directory_traversal', 'file_enumeration']
            },
            'T1071': {
                'name': 'Application Layer Protocol',
                'category': ThreatCategory.COMMAND_CONTROL,
                'severity': ThreatSeverity.HIGH,
                'indicators': ['suspicious_http_traffic', 'encoded_payloads', 'c2_communication']
            }
        }
    
    def _create_anomaly_detector(self):
        """Create anomaly detection model (simplified)"""
        class SimpleAnomalyDetector:
            def __init__(self):
                self.baseline_stats = {}
                self.threshold_multiplier = 2.5
            
            def fit(self, data):
                # Calculate baseline statistics
                if len(data) > 0:
                    self.baseline_stats = {
                        'mean': np.mean(data, axis=0),
                        'std': np.std(data, axis=0)
                    }
            
            def predict(self, features):
                if not self.baseline_stats:
                    return 0.5  # Neutral score if no baseline
                
                # Calculate z-scores
                z_scores = np.abs((features - self.baseline_stats['mean']) / 
                                (self.baseline_stats['std'] + 1e-8))
                
                # Anomaly score based on max z-score
                anomaly_score = min(np.max(z_scores) / self.threshold_multiplier, 1.0)
                return anomaly_score
        
        return SimpleAnomalyDetector()
    
    def _create_threat_classifier(self):
        """Create threat classification model (simplified)"""
        class SimpleThreatClassifier:
            def __init__(self):
                self.threat_signatures = {}
            
            def predict(self, features, patterns):
                scores = {}
                
                # Pattern-based classification
                for category, category_patterns in patterns.items():
                    score = 0
                    for pattern in category_patterns:
                        if hasattr(features, 'payload') and re.search(pattern, str(features.payload), re.IGNORECASE):
                            score += 1
                    scores[category] = score / len(category_patterns) if category_patterns else 0
                
                # Feature-based scoring
                if features.failed_attempts > 5:
                    scores['brute_force'] = scores.get('brute_force', 0) + 0.3
                
                if features.request_frequency > 10:
                    scores['reconnaissance'] = scores.get('reconnaissance', 0) + 0.2
                
                if features.suspicious_keywords > 3:
                    scores['command_injection'] = scores.get('command_injection', 0) + 0.4
                
                return scores
        
        return SimpleThreatClassifier()
    
    def _create_severity_predictor(self):
        """Create severity prediction model"""
        class SeverityPredictor:
            def predict(self, threat_scores, mitre_techniques):
                max_score = max(threat_scores.values()) if threat_scores else 0
                
                # Base severity on threat scores
                if max_score > 0.8:
                    base_severity = ThreatSeverity.CRITICAL
                elif max_score > 0.6:
                    base_severity = ThreatSeverity.HIGH
                elif max_score > 0.4:
                    base_severity = ThreatSeverity.MEDIUM
                elif max_score > 0.2:
                    base_severity = ThreatSeverity.LOW
                else:
                    base_severity = ThreatSeverity.INFO
                
                # Adjust based on MITRE techniques
                for technique in mitre_techniques:
                    if technique in ['T1190', 'T1110']:  # Critical techniques
                        base_severity = ThreatSeverity.CRITICAL
                        break
                
                return base_severity
        
        return SeverityPredictor()
    
    def _extract_network_features(self, interaction: Dict) -> Dict:
        """Extract network-level features"""
        return {
            'source_ip': interaction.get('source_ip', ''),
            'destination_port': interaction.get('port', 0),
            'protocol': interaction.get('protocol', ''),
            'packet_size': len(str(interaction.get('payload', ''))),
            'connection_duration': interaction.get('duration', 0)
        }
    
    def _extract_behavioral_features(self, interaction: Dict, history: List[Dict]) -> Dict:
        """Extract behavioral features from interaction history"""
        source_ip = interaction.get('source_ip', '')
        
        # Filter history for same source
        ip_history = [h for h in history if h.get('source_ip') == source_ip]
        
        # Calculate behavioral metrics
        request_frequency = len(ip_history) / max(1, len(history)) * 100
        unique_endpoints = len(set(h.get('endpoint', '') for h in ip_history))
        failed_attempts = sum(1 for h in ip_history if h.get('status_code', 200) >= 400)
        
        # Calculate entropy
        user_agent = interaction.get('user_agent', '')
        payload = str(interaction.get('payload', ''))
        
        user_agent_entropy = self._calculate_entropy(user_agent)
        payload_entropy = self._calculate_entropy(payload)
        
        return {
            'request_frequency': request_frequency,
            'unique_endpoints': unique_endpoints,
            'failed_attempts': failed_attempts,
            'user_agent_entropy': user_agent_entropy,
            'payload_entropy': payload_entropy
        }
    
    def _extract_temporal_features(self, interaction: Dict) -> Dict:
        """Extract temporal features"""
        timestamp = interaction.get('timestamp', datetime.now())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        return {
            'time_of_day': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'request_interval_variance': 0  # Simplified
        }
    
    def _extract_content_features(self, interaction: Dict) -> Dict:
        """Extract content-based features"""
        payload = str(interaction.get('payload', ''))
        endpoint = interaction.get('endpoint', '')
        user_agent = interaction.get('user_agent', '')
        
        content = f"{payload} {endpoint} {user_agent}".lower()
        
        # Count pattern matches
        sql_patterns = sum(1 for pattern in self.threat_patterns['sql_injection'] 
                          if re.search(pattern, content, re.IGNORECASE))
        
        xss_patterns = sum(1 for pattern in self.threat_patterns['xss_patterns'] 
                          if re.search(pattern, content, re.IGNORECASE))
        
        cmd_patterns = sum(1 for pattern in self.threat_patterns['command_injection'] 
                          if re.search(pattern, content, re.IGNORECASE))
        
        # Suspicious keywords
        suspicious_words = ['admin', 'root', 'password', 'exploit', 'hack', 'shell', 'cmd']
        suspicious_count = sum(1 for word in suspicious_words if word in content)
        
        return {
            'suspicious_keywords': suspicious_count,
            'sql_injection_patterns': sql_patterns,
            'xss_patterns': xss_patterns,
            'command_injection_patterns': cmd_patterns
        }
    
    def _extract_mitre_features(self, interaction: Dict) -> List[str]:
        """Extract MITRE ATT&CK technique indicators"""
        techniques = []
        payload = str(interaction.get('payload', '')).lower()
        endpoint = interaction.get('endpoint', '').lower()
        
        # T1595 - Active Scanning
        if any(word in payload + endpoint for word in ['scan', 'enum', 'probe']):
            techniques.append('T1595')
        
        # T1110 - Brute Force
        if interaction.get('failed_attempts', 0) > 3:
            techniques.append('T1110')
        
        # T1190 - Exploit Public-Facing Application
        if any(re.search(pattern, payload) for pattern in self.threat_patterns['sql_injection']):
            techniques.append('T1190')
        
        # T1083 - File and Directory Discovery
        if any(word in payload + endpoint for word in ['../', '/etc/', '/proc/', 'dir']):
            techniques.append('T1083')
        
        # T1071 - Application Layer Protocol
        if self._calculate_entropy(payload) > 4.0:  # High entropy suggests encoding
            techniques.append('T1071')
        
        return techniques
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text"""
        if not text:
            return 0
        
        # Count character frequencies
        char_counts = defaultdict(int)
        for char in text:
            char_counts[char] += 1
        
        # Calculate entropy
        text_len = len(text)
        entropy = 0
        for count in char_counts.values():
            probability = count / text_len
            entropy -= probability * np.log2(probability)
        
        return entropy
    
    def extract_features(self, interaction: Dict) -> ThreatFeatures:
        """Extract all features from an interaction"""
        # Get recent history for behavioral analysis
        recent_history = list(self.interaction_history)[-100:]
        
        # Extract feature categories
        network_features = self._extract_network_features(interaction)
        behavioral_features = self._extract_behavioral_features(interaction, recent_history)
        temporal_features = self._extract_temporal_features(interaction)
        content_features = self._extract_content_features(interaction)
        mitre_techniques = self._extract_mitre_features(interaction)
        
        # Combine all features
        return ThreatFeatures(
            **network_features,
            **behavioral_features,
            **temporal_features,
            **content_features,
            mitre_techniques=mitre_techniques
        )
    
    def classify_threat(self, interaction: Dict) -> Optional[ThreatClassification]:
        """Classify a potential threat from interaction data"""
        try:
            # Extract features
            features = self.extract_features(interaction)
            
            # Convert to numerical vector for ML models
            feature_vector = features.to_vector()
            
            # Anomaly detection
            anomaly_score = self.models['anomaly_detector'].predict(feature_vector)
            
            # Threat classification
            threat_scores = self.models['classifier'].predict(features, self.threat_patterns)
            
            # Determine if this is a threat
            max_threat_score = max(threat_scores.values()) if threat_scores else 0
            combined_score = (anomaly_score + max_threat_score) / 2
            
            if combined_score < 0.3:
                return None  # Not a threat
            
            # Determine threat category
            threat_category = max(threat_scores.items(), key=lambda x: x[1])[0] if threat_scores else 'unknown'
            
            # Map to MITRE ATT&CK category
            mitre_category = self._map_to_mitre_category(threat_category, features.mitre_techniques)
            
            # Predict severity
            severity = self.models['severity_predictor'].predict(threat_scores, features.mitre_techniques)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(mitre_category, severity, features)
            
            # Create threat classification
            threat_id = hashlib.md5(f"{interaction.get('source_ip', '')}{time.time()}".encode()).hexdigest()[:8]
            
            classification = ThreatClassification(
                threat_id=threat_id,
                category=mitre_category,
                severity=severity,
                confidence=combined_score,
                mitre_techniques=features.mitre_techniques,
                indicators=self._extract_indicators(features, threat_scores),
                recommended_actions=recommendations,
                timestamp=datetime.now(),
                source_ip=features.source_ip,
                target_service=interaction.get('service', 'unknown')
            )
            
            # Update statistics
            self.stats['threats_detected'] += 1
            self.threat_history.append(asdict(classification))
            
            return classification
            
        except Exception as e:
            print(f"Error in threat classification: {e}")
            return None
    
    def _map_to_mitre_category(self, threat_type: str, mitre_techniques: List[str]) -> ThreatCategory:
        """Map threat type to MITRE ATT&CK category"""
        mapping = {
            'sql_injection': ThreatCategory.PRIVILEGE_ESCALATION,
            'xss_patterns': ThreatCategory.PRIVILEGE_ESCALATION,
            'command_injection': ThreatCategory.PRIVILEGE_ESCALATION,
            'reconnaissance': ThreatCategory.RECONNAISSANCE,
            'brute_force': ThreatCategory.CREDENTIAL_ACCESS
        }
        
        # Check MITRE techniques first
        for technique in mitre_techniques:
            if technique in self.mitre_mapping:
                return self.mitre_mapping[technique]['category']
        
        return mapping.get(threat_type, ThreatCategory.DISCOVERY)
    
    def _extract_indicators(self, features: ThreatFeatures, threat_scores: Dict) -> List[str]:
        """Extract threat indicators"""
        indicators = []
        
        if features.failed_attempts > 3:
            indicators.append(f"Multiple failed attempts: {features.failed_attempts}")
        
        if features.request_frequency > 5:
            indicators.append(f"High request frequency: {features.request_frequency:.1f}/min")
        
        if features.sql_injection_patterns > 0:
            indicators.append(f"SQL injection patterns detected: {features.sql_injection_patterns}")
        
        if features.xss_patterns > 0:
            indicators.append(f"XSS patterns detected: {features.xss_patterns}")
        
        if features.command_injection_patterns > 0:
            indicators.append(f"Command injection patterns: {features.command_injection_patterns}")
        
        if features.payload_entropy > 5.0:
            indicators.append(f"High payload entropy: {features.payload_entropy:.2f}")
        
        return indicators
    
    def _generate_recommendations(self, category: ThreatCategory, severity: ThreatSeverity, 
                                features: ThreatFeatures) -> List[str]:
        """Generate recommended actions based on threat classification"""
        recommendations = []
        
        # Base recommendations by category
        category_recommendations = {
            ThreatCategory.RECONNAISSANCE: [
                "Monitor source IP for additional scanning activity",
                "Review firewall rules for unnecessary open ports",
                "Consider IP blocking if pattern continues"
            ],
            ThreatCategory.CREDENTIAL_ACCESS: [
                "Implement account lockout policies",
                "Enable multi-factor authentication",
                "Monitor for successful logins from this IP",
                "Review password policies"
            ],
            ThreatCategory.PRIVILEGE_ESCALATION: [
                "Immediately block source IP",
                "Review application security controls",
                "Patch vulnerable applications",
                "Implement Web Application Firewall rules"
            ]
        }
        
        recommendations.extend(category_recommendations.get(category, []))
        
        # Severity-based recommendations
        if severity in [ThreatSeverity.HIGH, ThreatSeverity.CRITICAL]:
            recommendations.extend([
                "Escalate to security team immediately",
                "Consider blocking source IP at network perimeter",
                "Review logs for successful exploitation attempts"
            ])
        
        # Feature-specific recommendations
        if features.mitre_techniques:
            recommendations.append(f"Review MITRE ATT&CK techniques: {', '.join(features.mitre_techniques)}")
        
        return recommendations
    
    def update_model(self, feedback: Dict):
        """Update models based on feedback (for continuous learning)"""
        # In production, this would retrain models with new data
        if feedback.get('false_positive'):
            self.stats['false_positives'] += 1
        
        # Update model accuracy
        total_feedback = self.stats['threats_detected']
        if total_feedback > 0:
            self.stats['model_accuracy'] = 1 - (self.stats['false_positives'] / total_feedback)
    
    def get_threat_statistics(self) -> Dict:
        """Get threat detection statistics"""
        return {
            **self.stats,
            'recent_threats': len(self.threat_history),
            'top_threat_categories': self._get_top_categories(),
            'top_mitre_techniques': self._get_top_mitre_techniques()
        }
    
    def _get_top_categories(self) -> Dict[str, int]:
        """Get top threat categories from recent history"""
        categories = defaultdict(int)
        for threat in self.threat_history:
            categories[threat['category']] += 1
        return dict(sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5])
    
    def _get_top_mitre_techniques(self) -> Dict[str, int]:
        """Get top MITRE techniques from recent history"""
        techniques = defaultdict(int)
        for threat in self.threat_history:
            for technique in threat.get('mitre_techniques', []):
                techniques[technique] += 1
        return dict(sorted(techniques.items(), key=lambda x: x[1], reverse=True)[:5])

# Global threat classifier instance
threat_classifier = AdvancedThreatClassifier() 