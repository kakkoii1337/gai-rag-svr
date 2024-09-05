from pydantic import BaseModel

class SplitDocRequestPydantic(BaseModel):
    DocumentId: str
    ChunkSize: int = 1000
    ChunkOverlap: int = 100