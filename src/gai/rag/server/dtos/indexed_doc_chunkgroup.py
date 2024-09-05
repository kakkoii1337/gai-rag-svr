from pydantic import BaseModel, Field
from typing import Optional
from uuid import uuid4

class IndexedDocChunkGroupPydantic(BaseModel):
    Id: str = Field(default_factory=uuid4)
    DocumentId: str  # Assuming DocumentId is also a UUID; adjust if it's not
    SplitAlgo: Optional[str] = None
    ChunkCount: int
    ChunkSize: int
    Overlap: int
    IsActive: bool = True
    ChunksDir: Optional[str] = None
