from pydantic import BaseModel
from typing import Optional, Dict
from datetime import datetime

class ContentItem(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    body: str
    metadata: Optional[Dict] = {}
    created_at: Optional[datetime] = None

class Feedback(BaseModel):
    content_id: str
    event: str  # 'view' | 'click' | 'skip'
    view_time: Optional[float] = 0.0
    user_id: Optional[str] = None
