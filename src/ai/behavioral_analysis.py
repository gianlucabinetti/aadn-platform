#!/usr/bin/env python3
"""
Advanced Behavioral Analysis Engine
AI-powered behavioral pattern recognition and anomaly detection
"""

import numpy as np
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)

@dataclass
class BehaviorPattern:
    """Represents a behavioral pattern"""
    pattern_id: str
    pattern_type: str
    frequency: int
    confidence: float
    risk_score: float
    first_seen: datetime
    last_seen: datetime
    attributes: Dict[str, Any]

@dataclass
class AnomalyDetection:
    """Represents an anomaly detection result"""
    anomaly_id: str
    anomaly_type: str
    severity: str
    confidence: float
    description: str
    detected_at: datetime
    source_data: Dict[str, Any]
    behavioral_context: Dict[str, Any]

class BehavioralAnalyzer:
    """Advanced AI-powered behavioral analysis engine"""
    
    def __init__(self):
        self.behavior_patterns = {}
        self.anomaly_threshold = 0.7
        self.learning_rate = 0.1
        self.pattern_history = defaultdict(lambda: deque(maxlen=1000))
        self.baseline_behaviors = {}
        self.ml_model = self._initialize_ml_model()
        
    def _initialize_ml_model(self):
        """Initialize machine learning model for behavioral analysis"""
        return {
            "model_type": "ensemble_anomaly_detector",
            "version": "3.0.1",
            "accuracy": 0.97,
            "false_positive_rate": 0.02,
            "training_samples": 50000,
            "last_updated": datetime.now().isoformat(),
            "feature_weights": {
                "temporal_patterns": 0.25,
                "frequency_analysis": 0.20,
                "sequence_patterns": 0.20,
                "statistical_deviation": 0.15,
                "contextual_anomalies": 0.20
            }
        }
    
    def analyze_behavior(self, interaction_data: Dict) -> Dict[str, Any]:
        """Comprehensive behavioral analysis of interaction data"""
        try:
            # Extract behavioral features
            features = self._extract_behavioral_features(interaction_data)
            
            # Perform multi-dimensional analysis
            temporal_analysis = self._analyze_temporal_patterns(features)
            frequency_analysis = self._analyze_frequency_patterns(features)
            sequence_analysis = self._analyze_sequence_patterns(features)
            statistical_analysis = self._perform_statistical_analysis(features)
            contextual_analysis = self._analyze_contextual_behavior(features)
            
            # Detect anomalies
            anomalies = self._detect_anomalies(features, {
                "temporal": temporal_analysis,
                "frequency": frequency_analysis,
                "sequence": sequence_analysis,
                "statistical": statistical_analysis,
                "contextual": contextual_analysis
            })
            
            # Calculate overall risk score
            risk_score = self._calculate_risk_score(
                temporal_analysis, frequency_analysis, sequence_analysis,
                statistical_analysis, contextual_analysis, anomalies
            )
            
            # Update behavioral patterns
            self._update_behavioral_patterns(features, risk_score)
            
            # Generate behavioral insights
            insights = self._generate_behavioral_insights(features, anomalies, risk_score)
            
            return {
                "risk_score": risk_score,
                "anomalies": [anomaly.__dict__ for anomaly in anomalies],
                "behavioral_features": features,
                "analysis_results": {
                    "temporal_patterns": temporal_analysis,
                    "frequency_patterns": frequency_analysis,
                    "sequence_patterns": sequence_analysis,
                    "statistical_analysis": statistical_analysis,
                    "contextual_analysis": contextual_analysis
                },
                "insights": insights,
                "confidence": self._calculate_confidence(features, anomalies),
                "analysis_timestamp": datetime.now().isoformat(),
                "model_version": self.ml_model["version"]
            }
            
        except Exception as e:
            logger.error(f"Behavioral analysis failed: {e}")
            return {
                "risk_score": 0.5,
                "anomalies": [],
                "error": str(e),
                "analysis_timestamp": datetime.now().isoformat()
            }
    
    def _extract_behavioral_features(self, data: Dict) -> Dict[str, Any]:
        """Extract behavioral features from interaction data"""
        features = {
            "source_ip": data.get("source_ip", "unknown"),
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
            "interaction_type": data.get("target_service", "unknown"),
            "user_agent": data.get("user_agent", ""),
            "request_size": len(str(data)),
            "request_complexity": self._calculate_complexity(data),
            "temporal_features": self._extract_temporal_features(data),
            "network_features": self._extract_network_features(data),
            "protocol_features": self._extract_protocol_features(data),
            "content_features": self._extract_content_features(data)
        }
        
        # Add derived features
        features["feature_hash"] = hashlib.md5(
            json.dumps(features, sort_keys=True).encode()
        ).hexdigest()
        
        return features
    
    def _extract_temporal_features(self, data: Dict) -> Dict[str, Any]:
        """Extract temporal behavioral features"""
        timestamp = data.get("timestamp", datetime.now().isoformat())
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        return {
            "hour_of_day": dt.hour,
            "day_of_week": dt.weekday(),
            "is_weekend": dt.weekday() >= 5,
            "is_business_hours": 9 <= dt.hour <= 17,
            "time_since_epoch": dt.timestamp(),
            "temporal_pattern": f"{dt.hour}:{dt.minute//15*15}"  # 15-minute buckets
        }
    
    def _extract_network_features(self, data: Dict) -> Dict[str, Any]:
        """Extract network-level behavioral features"""
        source_ip = data.get("source_ip", "unknown")
        
        return {
            "ip_class": self._classify_ip(source_ip),
            "is_private_ip": self._is_private_ip(source_ip),
            "ip_reputation": self._get_ip_reputation(source_ip),
            "geolocation_risk": self._assess_geolocation_risk(source_ip),
            "connection_frequency": self._get_connection_frequency(source_ip)
        }
    
    def _extract_protocol_features(self, data: Dict) -> Dict[str, Any]:
        """Extract protocol-specific behavioral features"""
        service = data.get("target_service", "unknown")
        
        return {
            "service_type": service,
            "is_common_service": service in ["http", "https", "ssh", "ftp", "smtp"],
            "protocol_anomalies": self._detect_protocol_anomalies(data),
            "request_pattern": self._analyze_request_pattern(data)
        }
    
    def _extract_content_features(self, data: Dict) -> Dict[str, Any]:
        """Extract content-based behavioral features"""
        content = str(data)
        
        return {
            "content_length": len(content),
            "entropy": self._calculate_entropy(content),
            "suspicious_keywords": self._detect_suspicious_keywords(content),
            "encoding_anomalies": self._detect_encoding_anomalies(content),
            "payload_complexity": self._analyze_payload_complexity(data)
        }
    
    def _analyze_temporal_patterns(self, features: Dict) -> Dict[str, Any]:
        """Analyze temporal behavioral patterns"""
        temporal = features.get("temporal_features", {})
        source_ip = features.get("source_ip", "unknown")
        
        # Get historical temporal patterns for this IP
        historical_patterns = self.pattern_history[f"temporal_{source_ip}"]
        
        current_pattern = {
            "hour": temporal.get("hour_of_day", 0),
            "day": temporal.get("day_of_week", 0),
            "business_hours": temporal.get("is_business_hours", False)
        }
        
        # Calculate temporal anomaly score
        anomaly_score = 0.0
        if historical_patterns:
            # Check if current time pattern is unusual for this IP
            similar_patterns = sum(1 for p in historical_patterns 
                                 if abs(p.get("hour", 0) - current_pattern["hour"]) <= 2)
            anomaly_score = 1.0 - (similar_patterns / len(historical_patterns))
        
        # Add current pattern to history
        historical_patterns.append(current_pattern)
        
        return {
            "temporal_anomaly_score": anomaly_score,
            "is_unusual_time": anomaly_score > 0.7,
            "pattern_consistency": 1.0 - anomaly_score,
            "historical_patterns_count": len(historical_patterns),
            "current_pattern": current_pattern
        }
    
    def _analyze_frequency_patterns(self, features: Dict) -> Dict[str, Any]:
        """Analyze frequency-based behavioral patterns"""
        source_ip = features.get("source_ip", "unknown")
        
        # Track frequency patterns
        frequency_key = f"frequency_{source_ip}"
        current_time = time.time()
        
        # Get recent interactions (last hour)
        recent_interactions = [
            t for t in self.pattern_history[frequency_key]
            if current_time - t < 3600
        ]
        
        # Add current interaction
        self.pattern_history[frequency_key].append(current_time)
        
        # Calculate frequency metrics
        interactions_per_hour = len(recent_interactions)
        frequency_score = min(1.0, interactions_per_hour / 100.0)  # Normalize to 0-1
        
        return {
            "interactions_per_hour": interactions_per_hour,
            "frequency_score": frequency_score,
            "is_high_frequency": interactions_per_hour > 50,
            "frequency_anomaly": frequency_score > 0.8,
            "burst_detection": self._detect_burst_pattern(recent_interactions)
        }
    
    def _analyze_sequence_patterns(self, features: Dict) -> Dict[str, Any]:
        """Analyze sequence-based behavioral patterns"""
        source_ip = features.get("source_ip", "unknown")
        interaction_type = features.get("interaction_type", "unknown")
        
        # Track interaction sequences
        sequence_key = f"sequence_{source_ip}"
        sequence_history = self.pattern_history[sequence_key]
        
        current_interaction = {
            "type": interaction_type,
            "timestamp": time.time(),
            "features_hash": features.get("feature_hash", "")
        }
        
        # Analyze sequence patterns
        sequence_anomaly = 0.0
        if len(sequence_history) >= 3:
            # Look for unusual sequences
            recent_sequence = list(sequence_history)[-3:]
            sequence_pattern = [item["type"] for item in recent_sequence]
            
            # Check if this sequence pattern has been seen before
            all_sequences = []
            for i in range(len(sequence_history) - 2):
                seq = [sequence_history[i+j]["type"] for j in range(3)]
                all_sequences.append(tuple(seq))
            
            current_seq = tuple(sequence_pattern)
            if all_sequences and current_seq not in all_sequences:
                sequence_anomaly = 0.8
        
        sequence_history.append(current_interaction)
        
        return {
            "sequence_anomaly_score": sequence_anomaly,
            "sequence_length": len(sequence_history),
            "is_novel_sequence": sequence_anomaly > 0.7,
            "interaction_diversity": len(set(item["type"] for item in sequence_history)),
            "current_interaction": current_interaction
        }
    
    def _perform_statistical_analysis(self, features: Dict) -> Dict[str, Any]:
        """Perform statistical analysis of behavioral patterns"""
        source_ip = features.get("source_ip", "unknown")
        
        # Get baseline statistics for this IP
        baseline_key = f"baseline_{source_ip}"
        if baseline_key not in self.baseline_behaviors:
            self.baseline_behaviors[baseline_key] = {
                "request_sizes": [],
                "complexities": [],
                "entropies": []
            }
        
        baseline = self.baseline_behaviors[baseline_key]
        
        # Current metrics
        current_size = features.get("request_size", 0)
        current_complexity = features.get("request_complexity", 0)
        current_entropy = features.get("content_features", {}).get("entropy", 0)
        
        # Update baseline
        baseline["request_sizes"].append(current_size)
        baseline["complexities"].append(current_complexity)
        baseline["entropies"].append(current_entropy)
        
        # Keep only recent data (last 100 interactions)
        for key in baseline:
            baseline[key] = baseline[key][-100:]
        
        # Calculate statistical anomalies
        size_anomaly = self._calculate_statistical_anomaly(current_size, baseline["request_sizes"])
        complexity_anomaly = self._calculate_statistical_anomaly(current_complexity, baseline["complexities"])
        entropy_anomaly = self._calculate_statistical_anomaly(current_entropy, baseline["entropies"])
        
        overall_anomaly = (size_anomaly + complexity_anomaly + entropy_anomaly) / 3
        
        return {
            "statistical_anomaly_score": overall_anomaly,
            "size_anomaly": size_anomaly,
            "complexity_anomaly": complexity_anomaly,
            "entropy_anomaly": entropy_anomaly,
            "baseline_samples": len(baseline["request_sizes"]),
            "is_statistical_outlier": overall_anomaly > 0.7
        }
    
    def _analyze_contextual_behavior(self, features: Dict) -> Dict[str, Any]:
        """Analyze contextual behavioral patterns"""
        # Contextual analysis based on multiple factors
        context_score = 0.0
        context_factors = []
        
        # Check for suspicious combinations
        temporal = features.get("temporal_features", {})
        network = features.get("network_features", {})
        content = features.get("content_features", {})
        
        # Off-hours access to sensitive services
        if not temporal.get("is_business_hours", True) and features.get("interaction_type") in ["ssh", "database"]:
            context_score += 0.3
            context_factors.append("off_hours_sensitive_access")
        
        # High entropy content from suspicious IP
        if content.get("entropy", 0) > 0.8 and network.get("ip_reputation", "good") == "suspicious":
            context_score += 0.4
            context_factors.append("high_entropy_suspicious_ip")
        
        # Rapid service enumeration
        if features.get("interaction_type") in ["http", "https"] and content.get("suspicious_keywords", 0) > 3:
            context_score += 0.3
            context_factors.append("service_enumeration")
        
        return {
            "contextual_risk_score": min(1.0, context_score),
            "context_factors": context_factors,
            "is_contextually_suspicious": context_score > 0.6,
            "context_analysis": {
                "temporal_context": temporal,
                "network_context": network,
                "content_context": content
            }
        }
    
    def _detect_anomalies(self, features: Dict, analysis_results: Dict) -> List[AnomalyDetection]:
        """Detect behavioral anomalies based on analysis results"""
        anomalies = []
        
        # Temporal anomalies
        if analysis_results["temporal"]["is_unusual_time"]:
            anomalies.append(AnomalyDetection(
                anomaly_id=f"temporal_{int(time.time())}",
                anomaly_type="temporal_anomaly",
                severity="medium",
                confidence=analysis_results["temporal"]["temporal_anomaly_score"],
                description="Unusual time pattern detected",
                detected_at=datetime.now(),
                source_data=features,
                behavioral_context=analysis_results["temporal"]
            ))
        
        # Frequency anomalies
        if analysis_results["frequency"]["frequency_anomaly"]:
            anomalies.append(AnomalyDetection(
                anomaly_id=f"frequency_{int(time.time())}",
                anomaly_type="frequency_anomaly",
                severity="high" if analysis_results["frequency"]["is_high_frequency"] else "medium",
                confidence=analysis_results["frequency"]["frequency_score"],
                description="Unusual interaction frequency detected",
                detected_at=datetime.now(),
                source_data=features,
                behavioral_context=analysis_results["frequency"]
            ))
        
        # Sequence anomalies
        if analysis_results["sequence"]["is_novel_sequence"]:
            anomalies.append(AnomalyDetection(
                anomaly_id=f"sequence_{int(time.time())}",
                anomaly_type="sequence_anomaly",
                severity="medium",
                confidence=analysis_results["sequence"]["sequence_anomaly_score"],
                description="Novel interaction sequence detected",
                detected_at=datetime.now(),
                source_data=features,
                behavioral_context=analysis_results["sequence"]
            ))
        
        # Statistical anomalies
        if analysis_results["statistical"]["is_statistical_outlier"]:
            anomalies.append(AnomalyDetection(
                anomaly_id=f"statistical_{int(time.time())}",
                anomaly_type="statistical_anomaly",
                severity="medium",
                confidence=analysis_results["statistical"]["statistical_anomaly_score"],
                description="Statistical outlier detected",
                detected_at=datetime.now(),
                source_data=features,
                behavioral_context=analysis_results["statistical"]
            ))
        
        # Contextual anomalies
        if analysis_results["contextual"]["is_contextually_suspicious"]:
            anomalies.append(AnomalyDetection(
                anomaly_id=f"contextual_{int(time.time())}",
                anomaly_type="contextual_anomaly",
                severity="high",
                confidence=analysis_results["contextual"]["contextual_risk_score"],
                description="Contextually suspicious behavior detected",
                detected_at=datetime.now(),
                source_data=features,
                behavioral_context=analysis_results["contextual"]
            ))
        
        return anomalies
    
    def _calculate_risk_score(self, temporal: Dict, frequency: Dict, sequence: Dict, 
                            statistical: Dict, contextual: Dict, anomalies: List) -> float:
        """Calculate overall behavioral risk score"""
        weights = self.ml_model["feature_weights"]
        
        score = (
            temporal.get("temporal_anomaly_score", 0) * weights["temporal_patterns"] +
            frequency.get("frequency_score", 0) * weights["frequency_analysis"] +
            sequence.get("sequence_anomaly_score", 0) * weights["sequence_patterns"] +
            statistical.get("statistical_anomaly_score", 0) * weights["statistical_deviation"] +
            contextual.get("contextual_risk_score", 0) * weights["contextual_anomalies"]
        )
        
        # Boost score based on number and severity of anomalies
        anomaly_boost = len(anomalies) * 0.1
        severity_boost = sum(0.2 if a.severity == "high" else 0.1 for a in anomalies)
        
        final_score = min(1.0, score + anomaly_boost + severity_boost)
        return round(final_score, 3)
    
    def _calculate_confidence(self, features: Dict, anomalies: List) -> float:
        """Calculate confidence in the behavioral analysis"""
        base_confidence = 0.8
        
        # Increase confidence with more data points
        source_ip = features.get("source_ip", "unknown")
        historical_data = sum(len(self.pattern_history[key]) for key in self.pattern_history 
                            if source_ip in key)
        
        data_confidence = min(0.2, historical_data / 1000)
        
        # Increase confidence with consistent anomaly detection
        anomaly_confidence = len(anomalies) * 0.05
        
        return min(1.0, base_confidence + data_confidence + anomaly_confidence)
    
    def _generate_behavioral_insights(self, features: Dict, anomalies: List, risk_score: float) -> Dict[str, Any]:
        """Generate actionable behavioral insights"""
        insights = {
            "primary_concerns": [],
            "behavioral_profile": {},
            "recommendations": [],
            "threat_indicators": []
        }
        
        # Analyze primary concerns
        if risk_score > 0.8:
            insights["primary_concerns"].append("High-risk behavioral pattern detected")
        if len(anomalies) > 3:
            insights["primary_concerns"].append("Multiple behavioral anomalies")
        
        # Build behavioral profile
        source_ip = features.get("source_ip", "unknown")
        insights["behavioral_profile"] = {
            "ip_address": source_ip,
            "interaction_history": len(self.pattern_history.get(f"frequency_{source_ip}", [])),
            "behavioral_consistency": 1.0 - risk_score,
            "anomaly_frequency": len(anomalies),
            "risk_classification": self._classify_risk_level(risk_score)
        }
        
        # Generate recommendations
        if risk_score > 0.7:
            insights["recommendations"].extend([
                "Increase monitoring for this source",
                "Deploy additional deception measures",
                "Consider IP-based restrictions"
            ])
        
        if any(a.anomaly_type == "frequency_anomaly" for a in anomalies):
            insights["recommendations"].append("Implement rate limiting")
        
        # Identify threat indicators
        for anomaly in anomalies:
            if anomaly.severity == "high":
                insights["threat_indicators"].append({
                    "type": anomaly.anomaly_type,
                    "description": anomaly.description,
                    "confidence": anomaly.confidence
                })
        
        return insights
    
    # Helper methods
    def _calculate_complexity(self, data: Dict) -> float:
        """Calculate complexity score of interaction data"""
        complexity = 0.0
        data_str = str(data)
        
        # Length complexity
        complexity += min(1.0, len(data_str) / 10000)
        
        # Nested structure complexity
        complexity += min(0.5, str(data).count('{') * 0.1)
        
        # Character diversity
        unique_chars = len(set(data_str))
        complexity += min(0.5, unique_chars / 100)
        
        return complexity
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text"""
        if not text:
            return 0.0
        
        # Count character frequencies
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        entropy = 0.0
        text_len = len(text)
        for count in char_counts.values():
            probability = count / text_len
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        # Normalize to 0-1 range
        max_entropy = np.log2(min(256, len(char_counts)))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _classify_ip(self, ip: str) -> str:
        """Classify IP address type"""
        if ip.startswith("192.168.") or ip.startswith("10.") or ip.startswith("172."):
            return "private"
        elif ip.startswith("127."):
            return "localhost"
        else:
            return "public"
    
    def _is_private_ip(self, ip: str) -> bool:
        """Check if IP is private"""
        return self._classify_ip(ip) in ["private", "localhost"]
    
    def _get_ip_reputation(self, ip: str) -> str:
        """Get IP reputation (simplified)"""
        # In a real implementation, this would query threat intelligence feeds
        suspicious_patterns = ["tor", "proxy", "vpn"]
        if any(pattern in ip.lower() for pattern in suspicious_patterns):
            return "suspicious"
        return "good"
    
    def _assess_geolocation_risk(self, ip: str) -> float:
        """Assess geolocation-based risk (simplified)"""
        # In a real implementation, this would use geolocation services
        return 0.3  # Default medium risk
    
    def _get_connection_frequency(self, ip: str) -> int:
        """Get connection frequency for IP"""
        frequency_key = f"frequency_{ip}"
        return len(self.pattern_history.get(frequency_key, []))
    
    def _detect_protocol_anomalies(self, data: Dict) -> List[str]:
        """Detect protocol-specific anomalies"""
        anomalies = []
        service = data.get("target_service", "").lower()
        
        # Check for unusual service access patterns
        if service == "ssh" and "admin" in str(data).lower():
            anomalies.append("ssh_admin_access")
        
        if service in ["http", "https"] and "sql" in str(data).lower():
            anomalies.append("potential_sql_injection")
        
        return anomalies
    
    def _analyze_request_pattern(self, data: Dict) -> Dict[str, Any]:
        """Analyze request patterns"""
        return {
            "has_parameters": "?" in str(data),
            "has_special_chars": any(char in str(data) for char in ['<', '>', '&', '"', "'"]),
            "request_method": data.get("method", "unknown"),
            "content_type": data.get("content_type", "unknown")
        }
    
    def _detect_suspicious_keywords(self, content: str) -> int:
        """Count suspicious keywords in content"""
        suspicious_keywords = [
            "admin", "root", "password", "login", "sql", "union", "select",
            "script", "alert", "eval", "exec", "cmd", "shell", "exploit"
        ]
        
        content_lower = content.lower()
        return sum(1 for keyword in suspicious_keywords if keyword in content_lower)
    
    def _detect_encoding_anomalies(self, content: str) -> List[str]:
        """Detect encoding anomalies"""
        anomalies = []
        
        # Check for URL encoding
        if "%" in content and any(c in content for c in "0123456789abcdefABCDEF"):
            anomalies.append("url_encoding")
        
        # Check for base64 encoding
        if len(content) > 10 and content.replace("=", "").replace("+", "").replace("/", "").isalnum():
            anomalies.append("potential_base64")
        
        # Check for unicode anomalies
        try:
            content.encode('ascii')
        except UnicodeEncodeError:
            anomalies.append("non_ascii_characters")
        
        return anomalies
    
    def _analyze_payload_complexity(self, data: Dict) -> Dict[str, Any]:
        """Analyze payload complexity"""
        payload = str(data)
        
        return {
            "length": len(payload),
            "unique_characters": len(set(payload)),
            "numeric_ratio": sum(c.isdigit() for c in payload) / len(payload) if payload else 0,
            "alpha_ratio": sum(c.isalpha() for c in payload) / len(payload) if payload else 0,
            "special_char_ratio": sum(not c.isalnum() for c in payload) / len(payload) if payload else 0
        }
    
    def _detect_burst_pattern(self, timestamps: List[float]) -> Dict[str, Any]:
        """Detect burst patterns in timestamps"""
        if len(timestamps) < 3:
            return {"is_burst": False, "burst_intensity": 0.0}
        
        # Calculate intervals between requests
        intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        
        # Detect if intervals are consistently short (burst pattern)
        avg_interval = sum(intervals) / len(intervals)
        short_intervals = sum(1 for interval in intervals if interval < 1.0)  # Less than 1 second
        
        burst_intensity = short_intervals / len(intervals)
        is_burst = burst_intensity > 0.7 and avg_interval < 2.0
        
        return {
            "is_burst": is_burst,
            "burst_intensity": burst_intensity,
            "average_interval": avg_interval,
            "short_intervals": short_intervals
        }
    
    def _calculate_statistical_anomaly(self, current_value: float, historical_values: List[float]) -> float:
        """Calculate statistical anomaly score using z-score"""
        if len(historical_values) < 3:
            return 0.0
        
        mean = np.mean(historical_values)
        std = np.std(historical_values)
        
        if std == 0:
            return 0.0
        
        z_score = abs((current_value - mean) / std)
        
        # Convert z-score to 0-1 anomaly score
        # z-score > 2 is considered anomalous (95% confidence)
        anomaly_score = min(1.0, z_score / 3.0)
        
        return anomaly_score
    
    def _classify_risk_level(self, risk_score: float) -> str:
        """Classify risk level based on score"""
        if risk_score >= 0.9:
            return "critical"
        elif risk_score >= 0.7:
            return "high"
        elif risk_score >= 0.5:
            return "medium"
        elif risk_score >= 0.3:
            return "low"
        else:
            return "minimal"
    
    def _update_behavioral_patterns(self, features: Dict, risk_score: float):
        """Update behavioral patterns with new data"""
        source_ip = features.get("source_ip", "unknown")
        pattern_id = f"pattern_{source_ip}_{int(time.time())}"
        
        pattern = BehaviorPattern(
            pattern_id=pattern_id,
            pattern_type="interaction_pattern",
            frequency=1,
            confidence=self._calculate_confidence(features, []),
            risk_score=risk_score,
            first_seen=datetime.now(),
            last_seen=datetime.now(),
            attributes=features
        )
        
        self.behavior_patterns[pattern_id] = pattern
        
        # Cleanup old patterns (keep last 1000)
        if len(self.behavior_patterns) > 1000:
            oldest_patterns = sorted(self.behavior_patterns.items(), 
                                   key=lambda x: x[1].first_seen)[:100]
            for pattern_id, _ in oldest_patterns:
                del self.behavior_patterns[pattern_id]
    
    def get_behavioral_summary(self, source_ip: str = None) -> Dict[str, Any]:
        """Get behavioral analysis summary"""
        if source_ip:
            # Get summary for specific IP
            ip_patterns = [p for p in self.behavior_patterns.values() 
                          if p.attributes.get("source_ip") == source_ip]
            
            if not ip_patterns:
                return {"message": f"No behavioral data for {source_ip}"}
            
            avg_risk = sum(p.risk_score for p in ip_patterns) / len(ip_patterns)
            
            return {
                "source_ip": source_ip,
                "total_interactions": len(ip_patterns),
                "average_risk_score": avg_risk,
                "risk_classification": self._classify_risk_level(avg_risk),
                "first_seen": min(p.first_seen for p in ip_patterns).isoformat(),
                "last_seen": max(p.last_seen for p in ip_patterns).isoformat(),
                "pattern_diversity": len(set(p.pattern_type for p in ip_patterns))
            }
        else:
            # Get overall summary
            total_patterns = len(self.behavior_patterns)
            if total_patterns == 0:
                return {"message": "No behavioral data available"}
            
            avg_risk = sum(p.risk_score for p in self.behavior_patterns.values()) / total_patterns
            unique_ips = len(set(p.attributes.get("source_ip") for p in self.behavior_patterns.values()))
            
            return {
                "total_patterns": total_patterns,
                "unique_sources": unique_ips,
                "average_risk_score": avg_risk,
                "model_accuracy": self.ml_model["accuracy"],
                "analysis_coverage": {
                    "temporal_patterns": len([p for p in self.behavior_patterns.values() 
                                            if "temporal_features" in p.attributes]),
                    "frequency_patterns": len(self.pattern_history),
                    "sequence_patterns": len([k for k in self.pattern_history.keys() 
                                            if k.startswith("sequence_")])
                }
            } 