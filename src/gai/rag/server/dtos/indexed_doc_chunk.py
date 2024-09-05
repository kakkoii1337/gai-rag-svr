from pydantic import BaseModel, ConfigDict, Field
from typing import Optional
from uuid import UUID

class IndexedDocChunkPydantic(BaseModel):
    Id: str = Field(...)  # The ellipsis here indicates a required field with no default value
    ChunkHash: str
    ChunkGroupId: str
    ByteSize: int
    IsDuplicate: bool
    IsIndexed: bool
    Content: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)

