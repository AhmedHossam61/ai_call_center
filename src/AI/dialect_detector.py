import asyncio
import json
from pydoc import text
import time
from typing import Tuple
from helpers.config import get_settings

settings = get_settings()

class DialectDetector:
    """
    LLM-based dialect detector using Gemini 2.5 Flash
    """
    
    DIALECT_PROMPT = """أنت خبير لغوي متخصص في اللهجات العربية. حلل النص التالي بعناية:

    النص: "{text}"

    حدد اللهجة من بين الخيارات التالية فقط:
    - egyptian  (مصري)  : عايز، إيه، دا، مش، باشا، عامل إيه
    - gulf      (خليجي) : شلونك، شسوي، تكفى، أبي، شنو، الحين
    - sudanese  (سوداني): كيف الحال، شنو، زول، قديرة، جاي منو، الله يسلمك، ما عارف
    
    أجب بصيغة JSON فقط:
    {{
        "dialect": "اسم اللهجة بالإنجليزية",
        "confidence": "رقم من 0.0 إلى 1.0",
        "reasoning": "سبب الاختيار والكلمات المفتاحية المكتشفة"
    }}"""

    KEYWORD_RULES = {
    "egyptian":  ["عايز", "إيه", " دا ", "مش ", "باشا", "إزيك", "عامل إيه", "فين"],
    "gulf":      ["شلونك", "تكفى", "الحين", "شسوي", "وايد", "أبغى", "يبيلي"],
    "sudanese":  ["زول", "قديرة", "شنو", "الله يسلمك", "جاي منو", "ما عارف شنو", "منو ده"],
}

    def __init__(self, model_tuple=None):
        """
        Args:
            model_tuple: Tuple of (genai_client, model_name)
        """
        if model_tuple:
            self.client, self.model_name = model_tuple
            self.model_available = True
        else:
            try:
                import google.genai as genai
                self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
                self.model_name = "gemini-2.0-flash"
                self.model_available = True
            except:
                self.model_available = False
        
        self.supported_dialects = ['egyptian', 'gulf','sudanese']
        # self.supported_dialects = ['egyptian', 'gulf', 'levantine', 'moroccan', 'sudanese', 'msa']
    def _fast_detect(self, text: str):
        """
        Cheap keyword scan. Returns (dialect, confidence) if 2+ keywords match,
        otherwise returns None to fall through to the Gemini LLM call.
        Requires at least 2 keyword hits to avoid false positives.
        """
        scores = {}
        for dialect, keywords in self.KEYWORD_RULES.items():
            hits = sum(1 for kw in keywords if kw in text)
            if hits > 0:
                scores[dialect] = hits

        if not scores:
            return None

        best_dialect = max(scores, key=scores.get)
        hit_count    = scores[best_dialect]

        if hit_count < 2:
            return None   # only 1 keyword hit — not confident enough, use LLM

        confidence = min(0.60 + hit_count * 0.08, 0.95)
        return best_dialect, confidence

    def detect(self, text: str) -> Tuple[str, float]:
        if not text or len(text.strip()) < 3:
            return 'msa', 0.2

        # Fast path: keyword scan — skips the Gemini API call entirely
        fast_result = self._fast_detect(text)
        if fast_result:
            print(f"⚡ Dialect fast-detected: {fast_result[0]} ({fast_result[1]:.0%})")
            return fast_result

        if not self.model_available:
            return 'msa', 0.5

        prompt = self.DIALECT_PROMPT.format(text=text)
        
        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                
                result_text = self._clean_json_response(response.text)
                data = json.loads(result_text)
                
                dialect = data.get('dialect', 'msa').lower()
                confidence = float(data.get('confidence', 0.5))
                
                if dialect not in self.supported_dialects:
                    dialect = 'msa'

                return dialect, confidence

            except Exception as e:
                if "429" in str(e):
                    time.sleep(1 * (attempt + 1))
                    continue
                print(f"⚠️ Dialect Detection Error: {e}")
                break

        return 'msa', 0.5

    async def detect_async(self, text: str):
        return await asyncio.to_thread(self.detect, text)

    def _clean_json_response(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```json"):
            text = text.split("```json")[1].split("```")[0]
        elif text.startswith("```"):
            text = text.split("```")[1].split("```")[0]
        return text.strip()