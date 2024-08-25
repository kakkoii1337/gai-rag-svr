from pydantic import BaseModel, Field
from typing import Optional
from uuid import uuid4

class IndexedDocumentChunkGroupPydantic(BaseModel):
    Id: str = Field(default_factory=uuid4)
    DocumentId: str  # Assuming DocumentId is also a UUID; adjust if it's not
    SplitAlgo: Optional[str] = None
    ChunkCount: int
    ChunkSize: int
    Overlap: int
    IsActive: bool = True
    ChunksDir: Optional[str] = None

    @staticmethod
    def from_dalc(orm):
        return IndexedDocumentChunkGroupPydantic(
            Id=orm.Id,
            DocumentId=orm.DocumentId,
            SplitAlgo=orm.SplitAlgo,
            ChunkCount=orm.ChunkCount,
            ChunkSize=orm.ChunkSize,
            Overlap=orm.Overlap,
            IsActive=orm.IsActive,
            ChunksDir=orm.ChunksDir
        )

