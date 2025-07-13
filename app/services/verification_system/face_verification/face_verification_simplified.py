import logging

from app.services.verification_system.api_manager.faceplusplus_manager import FacePlusPlusManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceVerification:
    def __init__(self):
        self.confidence_threshold = 80.0  # 80% confidence threshold
        self.fpp_manager = FacePlusPlusManager()
