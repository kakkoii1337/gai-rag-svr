from pydantic import BaseModel, ConfigDict, Field
from typing import Optional
from uuid import UUID

class IndexedDocumentChunkPydantic(BaseModel):
    Id: str = Field(...)  # The ellipsis here indicates a required field with no default value
    ChunkHash: str
    ChunkGroupId: str
    ByteSize: int
    IsDuplicate: bool
    IsIndexed: bool
    Content: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)

    @staticmethod
    def from_dalc(orm):
        """
        Create an instance of IndexedDocumentChunkPydantic from a DALC model instance.

        Args:
        orm (IndexedDocumentChunk): An instance of the DALC model.

        Returns:
        IndexedDocumentChunkPydantic: An instance of IndexedDocumentChunkPydantic filled with data from the DALC model.
        """
        try:
            return IndexedDocumentChunkPydantic(
                Id=orm.Id,
                ChunkHash=orm.ChunkHash,
                ChunkGroupId=orm.ChunkGroupId,
                ByteSize=orm.ByteSize,
                IsDuplicate=orm.IsDuplicate,
                IsIndexed=orm.IsIndexed,
                Content=orm.Content
            )
        except Exception as e:
            raise RuntimeError("Failed to convert from DALC to Pydantic model.") from e
