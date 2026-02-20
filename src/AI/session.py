class CallSession:

    def __init__(self, session_id):
        self.session_id = session_id
        self.detected_dialect = 'msa' 
        self.dialect_confidence = 0.0
        self.conversation_history = []
        self.dialect_locked = False
        self.lock_thresholds = {
            "egyptian":  0.70,
            "gulf":      0.60,   # overlaps with MSA — accept lower confidence
            "sudanese":  0.60,   # same reason as gulf
            "levantine": 0.65,
            "moroccan":  0.75,   # very distinct — require higher confidence
            "msa":       0.80,
        }
    
    @property
    def active_dialect(self) -> str:
        """
        Returns the current best-guess dialect whether locked or not.
        Use this instead of detected_dialect so the system uses the best
        available dialect even before confidence crosses the lock threshold.
        """
        return self.detected_dialect

    def lock_dialect(self, dialect, confidence):
        
        if self.dialect_locked:
            return True

        # REPLACE WITH THIS:
        threshold = self.lock_thresholds.get(dialect, 0.70)
        if confidence >= threshold:
            self.detected_dialect = dialect
            self.dialect_confidence = confidence
            self.dialect_locked = True
            print(f"🔒 Dialect LOCKED to: {dialect} ({confidence:.2f} confidence)")
            return True
        
        if confidence > self.dialect_confidence:
            self.detected_dialect = dialect
            self.dialect_confidence = confidence
            print(f"⏳ Dialect tracking: {dialect} ({confidence:.2f})")
        
        return False
    
    def add_interaction(self, user_text, assistant_text):
        self.conversation_history.append({
            'user': user_text,
            'assistant': assistant_text
        })
    
    def get_context(self, num_turns=3):
        return self.conversation_history[-num_turns:] if self.conversation_history else []
    
    def get_stats(self):
        return {
            'id': self.session_id,
            'dialect': self.detected_dialect,
            'is_locked': self.dialect_locked,
            'total_turns': len(self.conversation_history)
        }