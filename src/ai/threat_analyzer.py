"""
AADN AI Threat Analyzer
AI-powered threat analysis and classification system
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

from ..core.logging_config import get_ai_logger
from ..monitoring.interaction_logger import get_interactions

logger = get_ai_logger()


class ThreatLevel(str, Enum):
    """Threat severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AttackCategory(str, Enum):
    """Attack classification categories"""
    RECONNAISSANCE = "reconnaissance"
    BRUTE_FORCE = "brute_force"
    EXPLOITATION = "exploitation"
    MALWARE = "malware"
    LATERAL_MOVEMENT = "lateral_movement"
    DATA_EXFILTRATION = "data_exfiltration"
    PERSISTENCE = "persistence"
    UNKNOWN = "unknown"


@dataclass
class ThreatIndicator:
    """Individual threat indicator"""
    type: str
    value: str
    confidence: float
    source: str
    timestamp: datetime
    context: Dict[str, Any]


@dataclass
class ThreatAnalysis:
    """Complete threat analysis result"""
    threat_id: str
    level: ThreatLevel
    category: AttackCategory
    confidence: float
    source_ip: str
    target_decoys: List[str]
    indicators: List[ThreatIndicator]
    timeline: List[Dict[str, Any]]
    recommendations: List[str]
    mitre_techniques: List[str]
    created_at: datetime


class InteractionAnalyzer:
    """Analyzes individual interactions for threats"""
    
    def __init__(self):
        self.suspicious_patterns = {
            'sql_injection': [
                r"union\s+select",
                r"or\s+1\s*=\s*1",
                r"drop\s+table",
                r"exec\s*\(",
                r"script\s*>",
            ],
            'command_injection': [
                r";\s*cat\s+",
                r";\s*ls\s+",
                r";\s*wget\s+",
                r";\s*curl\s+",
                r"\|\s*nc\s+",
            ],
            'directory_traversal': [
                r"\.\./",
                r"\.\.\\",
                r"%2e%2e%2f",
                r"%2e%2e%5c",
            ],
            'credential_stuffing': [
                r"admin:admin",
                r"root:root",
                r"admin:password",
                r"test:test",
            ]
        }
        
        self.malicious_user_agents = [
            "sqlmap",
            "nikto",
            "nmap",
            "masscan",
            "zap",
            "burp",
            "metasploit"
        ]
    
    def analyze_interaction(self, interaction: Dict[str, Any]) -> List[ThreatIndicator]:
        """Analyze a single interaction for threats"""
        indicators = []
        
        # Analyze based on interaction type
        interaction_type = interaction.get('interaction_type', '')
        data = interaction.get('data', {})
        
        if 'http' in interaction_type:
            indicators.extend(self._analyze_http_interaction(data))
        elif 'ssh' in interaction_type:
            indicators.extend(self._analyze_ssh_interaction(data))
        elif 'ftp' in interaction_type:
            indicators.extend(self._analyze_ftp_interaction(data))
        elif 'mysql' in interaction_type:
            indicators.extend(self._analyze_mysql_interaction(data))
        
        # Common analysis for all interactions
        indicators.extend(self._analyze_common_patterns(interaction))
        
        return indicators
    
    def _analyze_http_interaction(self, data: Dict[str, Any]) -> List[ThreatIndicator]:
        """Analyze HTTP-specific threats"""
        indicators = []
        
        path = data.get('path', '')
        user_agent = data.get('headers', {}).get('User-Agent', '')
        method = data.get('method', '')
        
        # Check for malicious user agents
        for malicious_ua in self.malicious_user_agents:
            if malicious_ua.lower() in user_agent.lower():
                indicators.append(ThreatIndicator(
                    type="malicious_user_agent",
                    value=user_agent,
                    confidence=0.8,
                    source="http_headers",
                    timestamp=datetime.utcnow(),
                    context={"pattern": malicious_ua}
                ))
        
        # Check for suspicious paths
        for pattern_type, patterns in self.suspicious_patterns.items():
            for pattern in patterns:
                if re.search(pattern, path, re.IGNORECASE):
                    indicators.append(ThreatIndicator(
                        type=pattern_type,
                        value=path,
                        confidence=0.7,
                        source="http_path",
                        timestamp=datetime.utcnow(),
                        context={"pattern": pattern}
                    ))
        
        # Check for suspicious methods
        if method in ['PUT', 'DELETE', 'PATCH'] and '/admin' in path:
            indicators.append(ThreatIndicator(
                type="admin_modification_attempt",
                value=f"{method} {path}",
                confidence=0.6,
                source="http_method",
                timestamp=datetime.utcnow(),
                context={"method": method, "path": path}
            ))
        
        return indicators
    
    def _analyze_ssh_interaction(self, data: Dict[str, Any]) -> List[ThreatIndicator]:
        """Analyze SSH-specific threats"""
        indicators = []
        
        # Check for brute force patterns
        if 'attempts' in data:
            attempt_count = len(data['attempts'])
            if attempt_count > 3:
                indicators.append(ThreatIndicator(
                    type="ssh_brute_force",
                    value=str(attempt_count),
                    confidence=0.8,
                    source="ssh_auth",
                    timestamp=datetime.utcnow(),
                    context={"attempt_count": attempt_count}
                ))
        
        # Check client banner for known attack tools
        client_banner = data.get('client_banner', '')
        for tool in ['libssh', 'paramiko', 'pexpect']:
            if tool in client_banner.lower():
                indicators.append(ThreatIndicator(
                    type="automated_ssh_tool",
                    value=client_banner,
                    confidence=0.6,
                    source="ssh_banner",
                    timestamp=datetime.utcnow(),
                    context={"tool": tool}
                ))
        
        return indicators
    
    def _analyze_ftp_interaction(self, data: Dict[str, Any]) -> List[ThreatIndicator]:
        """Analyze FTP-specific threats"""
        indicators = []
        
        command = data.get('command', '').upper()
        
        # Check for suspicious FTP commands
        suspicious_commands = ['SITE EXEC', 'SITE CHMOD', 'DELE', 'RMD']
        if any(cmd in command for cmd in suspicious_commands):
            indicators.append(ThreatIndicator(
                type="suspicious_ftp_command",
                value=command,
                confidence=0.7,
                source="ftp_command",
                timestamp=datetime.utcnow(),
                context={"command": command}
            ))
        
        return indicators
    
    def _analyze_mysql_interaction(self, data: Dict[str, Any]) -> List[ThreatIndicator]:
        """Analyze MySQL-specific threats"""
        indicators = []
        
        username = data.get('username', '')
        
        # Check for common attack usernames
        attack_usernames = ['root', 'admin', 'test', 'user', 'mysql', 'db']
        if username.lower() in attack_usernames:
            indicators.append(ThreatIndicator(
                type="common_attack_username",
                value=username,
                confidence=0.5,
                source="mysql_auth",
                timestamp=datetime.utcnow(),
                context={"username": username}
            ))
        
        return indicators
    
    def _analyze_common_patterns(self, interaction: Dict[str, Any]) -> List[ThreatIndicator]:
        """Analyze common threat patterns across all interaction types"""
        indicators = []
        
        source_ip = interaction.get('source_ip', '')
        
        # Check for private IP ranges (potential lateral movement)
        if self._is_private_ip(source_ip):
            indicators.append(ThreatIndicator(
                type="internal_source",
                value=source_ip,
                confidence=0.3,
                source="network",
                timestamp=datetime.utcnow(),
                context={"ip_type": "private"}
            ))
        
        return indicators
    
    def _is_private_ip(self, ip: str) -> bool:
        """Check if IP is in private ranges"""
        try:
            parts = [int(x) for x in ip.split('.')]
            if len(parts) != 4:
                return False
            
            # 10.0.0.0/8
            if parts[0] == 10:
                return True
            # 172.16.0.0/12
            if parts[0] == 172 and 16 <= parts[1] <= 31:
                return True
            # 192.168.0.0/16
            if parts[0] == 192 and parts[1] == 168:
                return True
            
        except (ValueError, IndexError):
            pass
        
        return False


class BehaviorAnalyzer:
    """Analyzes attacker behavior patterns"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.clustering_model = DBSCAN(eps=0.3, min_samples=2)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
    def analyze_attack_patterns(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in attack behavior"""
        if not interactions:
            return {}
        
        # Group interactions by source IP
        ip_groups = {}
        for interaction in interactions:
            source_ip = interaction.get('source_ip', 'unknown')
            if source_ip not in ip_groups:
                ip_groups[source_ip] = []
            ip_groups[source_ip].append(interaction)
        
        patterns = {}
        
        for source_ip, ip_interactions in ip_groups.items():
            patterns[source_ip] = {
                'total_interactions': len(ip_interactions),
                'unique_decoys': len(set(i.get('decoy_id', '') for i in ip_interactions)),
                'interaction_types': list(set(i.get('interaction_type', '') for i in ip_interactions)),
                'time_span': self._calculate_time_span(ip_interactions),
                'attack_velocity': self._calculate_attack_velocity(ip_interactions),
                'target_diversity': self._calculate_target_diversity(ip_interactions),
            }
        
        return patterns
    
    def detect_coordinated_attacks(self, interactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect coordinated attacks from multiple sources"""
        coordinated_attacks = []
        
        # Group by time windows (5-minute windows)
        time_windows = {}
        for interaction in interactions:
            timestamp = interaction.get('timestamp', datetime.utcnow())
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            window = timestamp.replace(second=0, microsecond=0)
            window = window.replace(minute=(window.minute // 5) * 5)
            
            if window not in time_windows:
                time_windows[window] = []
            time_windows[window].append(interaction)
        
        # Analyze each time window for coordination
        for window, window_interactions in time_windows.items():
            if len(window_interactions) < 3:  # Need at least 3 interactions
                continue
            
            unique_ips = set(i.get('source_ip', '') for i in window_interactions)
            if len(unique_ips) >= 2:  # Multiple source IPs
                coordinated_attacks.append({
                    'window': window,
                    'source_ips': list(unique_ips),
                    'interaction_count': len(window_interactions),
                    'target_decoys': list(set(i.get('decoy_id', '') for i in window_interactions)),
                    'coordination_score': self._calculate_coordination_score(window_interactions)
                })
        
        return coordinated_attacks
    
    def _calculate_time_span(self, interactions: List[Dict[str, Any]]) -> float:
        """Calculate time span of interactions in seconds"""
        if len(interactions) < 2:
            return 0.0
        
        timestamps = []
        for interaction in interactions:
            timestamp = interaction.get('timestamp', datetime.utcnow())
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            timestamps.append(timestamp)
        
        timestamps.sort()
        return (timestamps[-1] - timestamps[0]).total_seconds()
    
    def _calculate_attack_velocity(self, interactions: List[Dict[str, Any]]) -> float:
        """Calculate attack velocity (interactions per minute)"""
        time_span = self._calculate_time_span(interactions)
        if time_span == 0:
            return 0.0
        
        return len(interactions) / (time_span / 60)  # interactions per minute
    
    def _calculate_target_diversity(self, interactions: List[Dict[str, Any]]) -> float:
        """Calculate diversity of targets (0-1 scale)"""
        unique_decoys = set(i.get('decoy_id', '') for i in interactions)
        unique_types = set(i.get('interaction_type', '') for i in interactions)
        
        # Simple diversity score based on unique targets and types
        return min(1.0, (len(unique_decoys) + len(unique_types)) / 10)
    
    def _calculate_coordination_score(self, interactions: List[Dict[str, Any]]) -> float:
        """Calculate coordination score for a group of interactions"""
        # Factors: timing similarity, target similarity, technique similarity
        
        # Timing similarity (interactions within short time windows)
        timestamps = []
        for interaction in interactions:
            timestamp = interaction.get('timestamp', datetime.utcnow())
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            timestamps.append(timestamp)
        
        timestamps.sort()
        timing_score = 0.0
        if len(timestamps) > 1:
            max_gap = max((timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1))
            timing_score = max(0, 1 - (max_gap / 300))  # 5-minute window
        
        # Target similarity
        unique_decoys = set(i.get('decoy_id', '') for i in interactions)
        target_score = 1.0 / len(unique_decoys) if unique_decoys else 0.0
        
        # Technique similarity
        unique_types = set(i.get('interaction_type', '') for i in interactions)
        technique_score = 1.0 / len(unique_types) if unique_types else 0.0
        
        return (timing_score + target_score + technique_score) / 3


class ThreatAnalyzer:
    """Main threat analysis engine"""
    
    def __init__(self):
        self.interaction_analyzer = InteractionAnalyzer()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.threat_cache = {}
        
    async def analyze_real_time(self, interaction: Dict[str, Any]) -> Optional[ThreatAnalysis]:
        """Analyze a single interaction in real-time"""
        try:
            indicators = self.interaction_analyzer.analyze_interaction(interaction)
            
            if not indicators:
                return None
            
            # Calculate overall threat level
            threat_level = self._calculate_threat_level(indicators)
            category = self._classify_attack_category(indicators)
            confidence = self._calculate_confidence(indicators)
            
            # Get historical context for this source IP
            source_ip = interaction.get('source_ip', '')
            historical_interactions = await self._get_historical_interactions(source_ip)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(indicators, historical_interactions)
            
            # Map to MITRE ATT&CK techniques
            mitre_techniques = self._map_to_mitre(indicators)
            
            threat_analysis = ThreatAnalysis(
                threat_id=f"threat_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{source_ip.replace('.', '_')}",
                level=threat_level,
                category=category,
                confidence=confidence,
                source_ip=source_ip,
                target_decoys=[interaction.get('decoy_id', '')],
                indicators=indicators,
                timeline=[{
                    'timestamp': interaction.get('timestamp', datetime.utcnow()),
                    'event': 'interaction_detected',
                    'details': interaction
                }],
                recommendations=recommendations,
                mitre_techniques=mitre_techniques,
                created_at=datetime.utcnow()
            )
            
            # Cache the threat analysis
            self.threat_cache[threat_analysis.threat_id] = threat_analysis
            
            return threat_analysis
            
        except Exception as e:
            logger.error(f"Error in real-time threat analysis: {e}")
            return None
    
    async def analyze_batch(self, time_window: timedelta = timedelta(hours=1)) -> List[ThreatAnalysis]:
        """Analyze interactions in batch for pattern detection"""
        try:
            # Get recent interactions
            end_time = datetime.utcnow()
            start_time = end_time - time_window
            
            interactions = await get_interactions(
                start_time=start_time,
                end_time=end_time,
                limit=1000
            )
            
            if not interactions:
                return []
            
            # Analyze behavior patterns
            patterns = self.behavior_analyzer.analyze_attack_patterns(interactions)
            coordinated_attacks = self.behavior_analyzer.detect_coordinated_attacks(interactions)
            
            threat_analyses = []
            
            # Generate threat analyses for coordinated attacks
            for attack in coordinated_attacks:
                if attack['coordination_score'] > 0.5:  # High coordination threshold
                    threat_analysis = ThreatAnalysis(
                        threat_id=f"coordinated_{attack['window'].strftime('%Y%m%d_%H%M%S')}",
                        level=ThreatLevel.HIGH,
                        category=AttackCategory.RECONNAISSANCE,
                        confidence=attack['coordination_score'],
                        source_ip=','.join(attack['source_ips']),
                        target_decoys=attack['target_decoys'],
                        indicators=[],
                        timeline=[],
                        recommendations=[
                            "Block coordinated source IPs",
                            "Increase monitoring on targeted decoys",
                            "Review network segmentation"
                        ],
                        mitre_techniques=["T1595.001"],  # Active Scanning: Scanning IP Blocks
                        created_at=datetime.utcnow()
                    )
                    threat_analyses.append(threat_analysis)
            
            return threat_analyses
            
        except Exception as e:
            logger.error(f"Error in batch threat analysis: {e}")
            return []
    
    def _calculate_threat_level(self, indicators: List[ThreatIndicator]) -> ThreatLevel:
        """Calculate overall threat level from indicators"""
        if not indicators:
            return ThreatLevel.LOW
        
        max_confidence = max(indicator.confidence for indicator in indicators)
        indicator_count = len(indicators)
        
        # High-confidence indicators or many indicators = higher threat
        if max_confidence > 0.8 or indicator_count > 5:
            return ThreatLevel.HIGH
        elif max_confidence > 0.6 or indicator_count > 3:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def _classify_attack_category(self, indicators: List[ThreatIndicator]) -> AttackCategory:
        """Classify attack category based on indicators"""
        indicator_types = [indicator.type for indicator in indicators]
        
        if any('brute_force' in t for t in indicator_types):
            return AttackCategory.BRUTE_FORCE
        elif any('injection' in t for t in indicator_types):
            return AttackCategory.EXPLOITATION
        elif any('malicious' in t for t in indicator_types):
            return AttackCategory.MALWARE
        elif any('traversal' in t for t in indicator_types):
            return AttackCategory.EXPLOITATION
        else:
            return AttackCategory.RECONNAISSANCE
    
    def _calculate_confidence(self, indicators: List[ThreatIndicator]) -> float:
        """Calculate overall confidence score"""
        if not indicators:
            return 0.0
        
        # Weighted average of indicator confidences
        total_confidence = sum(indicator.confidence for indicator in indicators)
        return min(1.0, total_confidence / len(indicators))
    
    def _generate_recommendations(
        self,
        indicators: List[ThreatIndicator],
        historical_interactions: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        indicator_types = [indicator.type for indicator in indicators]
        
        if any('brute_force' in t for t in indicator_types):
            recommendations.append("Implement rate limiting on authentication endpoints")
            recommendations.append("Consider IP-based blocking for repeated failures")
        
        if any('injection' in t for t in indicator_types):
            recommendations.append("Review input validation and sanitization")
            recommendations.append("Implement Web Application Firewall (WAF)")
        
        if any('malicious_user_agent' in t for t in indicator_types):
            recommendations.append("Block known attack tool user agents")
            recommendations.append("Implement bot detection mechanisms")
        
        if len(historical_interactions) > 10:
            recommendations.append("Source IP shows persistent attack behavior - consider blocking")
        
        return recommendations
    
    def _map_to_mitre(self, indicators: List[ThreatIndicator]) -> List[str]:
        """Map indicators to MITRE ATT&CK techniques"""
        techniques = []
        indicator_types = [indicator.type for indicator in indicators]
        
        mitre_mapping = {
            'brute_force': ['T1110'],  # Brute Force
            'sql_injection': ['T1190'],  # Exploit Public-Facing Application
            'command_injection': ['T1190'],  # Exploit Public-Facing Application
            'directory_traversal': ['T1190'],  # Exploit Public-Facing Application
            'malicious_user_agent': ['T1595'],  # Active Scanning
            'reconnaissance': ['T1595'],  # Active Scanning
        }
        
        for indicator_type in indicator_types:
            for pattern, technique_ids in mitre_mapping.items():
                if pattern in indicator_type:
                    techniques.extend(technique_ids)
        
        return list(set(techniques))  # Remove duplicates
    
    async def _get_historical_interactions(self, source_ip: str) -> List[Dict[str, Any]]:
        """Get historical interactions for a source IP"""
        try:
            # Get interactions from the last 24 hours
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=24)
            
            interactions = await get_interactions(
                start_time=start_time,
                end_time=end_time,
                source_ip=source_ip,
                limit=100
            )
            
            return interactions
            
        except Exception as e:
            logger.error(f"Error getting historical interactions: {e}")
            return []


# Global threat analyzer instance
threat_analyzer = ThreatAnalyzer()


async def analyze_interaction(interaction: Dict[str, Any]) -> Optional[ThreatAnalysis]:
    """Analyze a single interaction for threats"""
    return await threat_analyzer.analyze_real_time(interaction)


async def analyze_batch_threats(time_window: timedelta = timedelta(hours=1)) -> List[ThreatAnalysis]:
    """Analyze recent interactions for threat patterns"""
    return await threat_analyzer.analyze_batch(time_window) 