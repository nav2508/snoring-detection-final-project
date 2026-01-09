from sqlalchemy import Column, Integer, Float, DateTime
from datetime import datetime
from database import Base


class SnoringSession(Base):
    __tablename__ = "snoring_sessions"

    id = Column(Integer, primary_key=True, index=True)

    audio_duration = Column(Float)
    snoring_ratio = Column(Float)
    quiet_minutes = Column(Float)

    nudges_sent = Column(Integer)

    created_at = Column(DateTime, default=datetime.utcnow)
