import cv2
import numpy as np
import pytesseract
import logging
import re
from PIL import Image
from typing import Optional 
import io

logger = logging.getLogger(__name__)

class OCRManager:
    def __init__(self):
        # Configure tesseract path if needed (Windows)
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        pass
    
    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """Preprocess image for better OCR results"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply threshold to get better contrast
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return thresh
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return None
    
    def extract_text_from_image(self, image_data: bytes) -> str:
        """Extract text from image using OCR"""
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_data)
            if processed_image is None:
                return ""
            
            # Configure OCR
            config = '--oem 3 --psm 6'
            
            # Extract text
            text = pytesseract.image_to_string(processed_image, config=config, lang='eng')
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from image: {str(e)}")
            return ""
    
    def extract_name_from_nid_passport(self, image_data: bytes) -> Optional[str]:
        """Extract name from NID/Passport image"""
        try:
            # Extract all text from image
            text = self.extract_text_from_image(image_data)
            
            if not text:
                return None
            
            # Common patterns for name extraction from NID/Passport
            name_patterns = [
                r'Name[:\s]*([A-Za-z\s]+)',
                r'NAME[:\s]*([A-Za-z\s]+)',
                r'নাম[:\s]*([A-Za-z\s]+)',  # Bengali for "Name"
                r'Full Name[:\s]*([A-Za-z\s]+)',
                r'FULL NAME[:\s]*([A-Za-z\s]+)',
                r'Holder[:\s]*([A-Za-z\s]+)',
                r'HOLDER[:\s]*([A-Za-z\s]+)',
            ]
            
            # Try each pattern
            for pattern in name_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    name = match.group(1).strip()
                    # Clean up the name (remove extra spaces, numbers, etc.)
                    name = ' '.join(name.split())
                    if len(name) > 2 and name.replace(' ', '').isalpha():
                        return name
            
            # If no pattern matches, try to find the longest alphabetic sequence
            # that might be a name (fallback method)
            words = text.split()
            potential_names = []
            
            for i, word in enumerate(words):
                if word.replace(' ', '').isalpha() and len(word) > 2:
                    # Check if next 1-2 words are also alphabetic (likely full name)
                    name_parts = [word]
                    for j in range(i + 1, min(i + 3, len(words))):
                        if words[j].replace(' ', '').isalpha():
                            name_parts.append(words[j])
                        else:
                            break
                    
                    if len(name_parts) >= 2:  # At least first and last name
                        potential_names.append(' '.join(name_parts))
            
            if potential_names:
                # Return the longest potential name
                return max(potential_names, key=len)
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting name from document: {str(e)}")
            return None