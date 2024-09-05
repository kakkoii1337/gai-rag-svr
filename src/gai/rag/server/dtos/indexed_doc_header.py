from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime,date
from gai.rag.server.dtos.indexed_doc_chunkgroup import IndexedDocChunkGroupPydantic

class IndexedDocHeaderPydantic(BaseModel):
    Id: str = Field(...)
    CollectionName: str
    ByteSize: int
    FileName: Optional[str] = None
    FileType: Optional[str] = None
    Source: Optional[str] = None
    Abstract: Optional[str] = None
    Authors: Optional[str] = None
    Title: Optional[str] = None
    Publisher: Optional[str] = None
    PublishedDate: Optional[date] = None
    Comments: Optional[str] = None
    Keywords: Optional[str] = None
    CreatedAt: datetime
    UpdatedAt: datetime
    IsActive: bool = True
    ChunkGroups: Optional[List[IndexedDocChunkGroupPydantic]] = None

