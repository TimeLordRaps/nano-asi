"""Advanced telemetry and logging system for NanoASI."""

import logging
from typing import Dict, Any
import time
import json

class TelemetryLogger:
    """Advanced telemetry and logging system."""
    
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger('NanoASI')
        self.logger.setLevel(log_level)
        
        # JSON log handler for structured logging
        json_handler = logging.FileHandler('nano_asi_telemetry.json')
        json_handler.setFormatter(JsonFormatter())
        self.logger.addHandler(json_handler)
    
    def log_event(
        self, 
        event_type: str, 
        data: Dict[str, Any], 
        level: int = logging.INFO
    ):
        """Log a structured event with telemetry."""
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'data': data
        }
        
        self.logger.log(level, json.dumps(event))

class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    def format(self, record):
        return record.msg
