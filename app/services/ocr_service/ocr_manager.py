import cv2
import numpy as np
import pytesseract
import re
from PIL import Image
import io
from typing import Optional, Dict

class OCRManager:
    """
    Minimal OCR manager for extracting name from NID/Passport images.
    For best results, consider using Google Cloud Vision or AWS Textract for production.
    """
    def extract_text_from_image(self, image_data: bytes) -> str:
        image = Image.open(io.BytesIO(image_data))
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # Resize for better OCR
        h, w = img.shape[:2]
        if w < 1000:
            scale = 1000 / w
            img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, config='--oem 3 --psm 6', lang='eng')
        return text

    def extract_name(self, image_data: bytes) -> Optional[str]:
        text = self.extract_text_from_image(image_data)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Skip machine-readable zone (MRZ) lines that start with P< or contain <<<
        lines = [line for line in lines if not (line.startswith('P<') or '<<<' in line)]
        
        # 1. Look for lines following "Given Names" labels
        for i, line in enumerate(lines):
            if re.search(r'Given Names?[/\s]', line, re.IGNORECASE):
                # Check next line for the actual name
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    # Remove common prefixes like "A ", "= ", "~ "
                    cleaned = re.sub(r'^[A-Z\s=~\-]*\s*', '', next_line)
                    if re.match(r'^[A-Z][A-Z\s]+$', cleaned) and len(cleaned) >= 3:
                        return cleaned.strip()
        
        # 1.5. Look for names on lines that start with single letters followed by uppercase names
        for line in lines:
            # Handle lines like "A POMICHELLE :" 
            match = re.search(r'^[A-Z]\s+([A-Z][A-Z\s]+)', line)
            if match:
                name = match.group(1).strip()
                # Remove trailing colons and clean up
                name = re.sub(r'[:\s]+$', '', name)
                if len(name) >= 4 and not any(word in name.upper() for word in ['PASSPORT', 'CARD', 'DOCUMENT']):
                    return name
        
        # 2. Look for name patterns with common prefixes (MD., MR., etc.)
        for line in lines:
            # Handle lines like "= MD. FARHAD BILL" or "~ SAYEDFUADALLABIB"
            cleaned = re.sub(r'^[=~\-\s]*', '', line).strip()
            if re.match(r'^(MD\.|MR\.|MRS\.|MS\.|DR\.|PROF\.|ENGR\.)\s+[A-Z\s]+$', cleaned):
                return cleaned
        
        # 3. Look for single long uppercase names (like SAYEDFUADALLABIB) and split them
        ignore_words = {'SPECIMEN', 'VZOR', 'SAMPLE', 'CARD', 'REPUBLIC', 'IDENTIFICATION', 'DOCUMENT', 'NATIONAL', 'ID', 'BIRTH', 'SIGNATURE', 'PASSPORT'}
        for line in lines:
            if any(word in line.upper() for word in ignore_words):
                continue
            cleaned = re.sub(r'^[=~\-\s]*', '', line).strip()
            if re.match(r'^[A-Z]{6,}$', cleaned):  # Single long uppercase word
                # Try to split merged names intelligently
                split_name = self.split_merged_name(cleaned)
                return split_name if split_name else cleaned
        
        # 4. Look for multi-word uppercase names
        for line in lines:
            if any(word in line.upper() for word in ignore_words):
                continue
            cleaned = re.sub(r'^[=~\-\s]*', '', line).strip()
            words = cleaned.split()
            if len(words) >= 2 and all(w.isupper() and len(w) >= 2 for w in words):
                return cleaned
        
        return None

    def extract_all_fields(self, image_data: bytes) -> Dict[str, Optional[str]]:
        text = self.extract_text_from_image(image_data)
        name = self.extract_name(image_data)
        return {'name': name, 'raw_text': text}

    def split_merged_name(self, merged_name: str) -> Optional[str]:
        """Split a merged uppercase name like SAYEDFUADALLABIB into SAYED FUAD AL LABIB"""
        if len(merged_name) < 6:
            return None
        
        # Common name patterns and prefixes to help with splitting
        common_prefixes = ['AL', 'EL', 'DE', 'VAN', 'VON', 'MC', 'MAC']
        common_names = [
            'AHMED', 'MOHAMMED', 'MOHAMMAD', 'ABDUL', 'ABDEL', 'SAYED', 'MOHAMED',
            'HASSAN', 'HUSSAIN', 'RAHMAN', 'KARIM', 'RAHIM', 'RASHID', 'MAHMUD',
            'FUAD', 'FARHAD', 'LABIB', 'HABIB', 'KHAN', 'ISLAM', 'HAQUE', 'ALAM'
        ]
        
        # Try to find known name patterns
        result_parts = []
        remaining = merged_name
        
        while remaining and len(remaining) >= 3:
            found_match = False
            
            # Check for common prefixes first (AL, EL, etc.)
            for prefix in common_prefixes:
                if remaining.startswith(prefix) and len(remaining) > len(prefix):
                    result_parts.append(prefix)
                    remaining = remaining[len(prefix):]
                    found_match = True
                    break
            
            if found_match:
                continue
            
            # Check for common names
            for name in sorted(common_names, key=len, reverse=True):  # Check longer names first
                if remaining.startswith(name):
                    result_parts.append(name)
                    remaining = remaining[len(name):]
                    found_match = True
                    break
            
            if not found_match:
                # If no pattern found, try to split at logical points
                # Look for consonant clusters or vowel patterns
                if len(remaining) >= 4:
                    # Simple heuristic: if we have at least 4 chars, take first 3-5 as a word
                    for split_len in [5, 4, 3]:
                        if len(remaining) >= split_len:
                            word = remaining[:split_len]
                            result_parts.append(word)
                            remaining = remaining[split_len:]
                            found_match = True
                            break
                
                if not found_match:
                    # Add remaining as one word
                    result_parts.append(remaining)
                    break
        
        if len(result_parts) >= 2:
            return ' '.join(result_parts)
        
        return None