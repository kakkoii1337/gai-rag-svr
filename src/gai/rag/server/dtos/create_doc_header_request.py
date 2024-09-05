from pydantic import BaseModel
from typing import Optional
class CreateDocHeaderRequestPydantic(BaseModel):
    Id: Optional[str] = None
    CollectionName: Optional[str]
    FilePath: Optional[str]
    FileType: Optional[str] = None
    Source: Optional[str] = None
    ByteSize: Optional[int] = None
    Title: Optional[str] = None
    Abstract: Optional[str] = None
    Authors: Optional[str] = None
    Publisher: Optional[str] = None
    PublishedDate: Optional[str] = None
    Comments: Optional[str] = None
    Keywords: Optional[str] = None
    ChunkGroupId: Optional[str] = None
    ChunkSize: Optional[int] = None
    ChunkOverlap: Optional[int] = None
    ChunkCount: Optional[int] = None    